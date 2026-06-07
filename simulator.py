"""SimpleFAIR: a lightweight wrapper around FaIR with GMST nudging and ECS control."""

import copy
import os
import warnings

import numpy as np
import pandas as pd

from fair import FAIR
from fair.constants import SPECIES_AXIS, TIME_AXIS
from fair.earth_params import earth_radius, seconds_per_year
from fair.energy_balance_model import (
    calculate_toa_imbalance_postrun,
    step_temperature,
)
from fair.forcing.aerosol.erfaci import logsum
from fair.forcing.aerosol.erfari import calculate_erfari_forcing
from fair.forcing.ghg import meinshausen2020
from fair.forcing.minor import calculate_linear_forcing
from fair.forcing.ozone import thornhill2021
from fair.gas_cycle import calculate_alpha
from fair.gas_cycle.ch4_lifetime import calculate_alpha_ch4
from fair.gas_cycle.eesc import calculate_eesc
from fair.gas_cycle.forward import step_concentration
from fair.gas_cycle.inverse import unstep_concentration
from fair.interface import fill, initialise
from fair.io import read_properties

HERE = os.path.dirname(os.path.realpath(__file__))
DEFAULT_EBM_CONFIG = os.path.join(HERE, "data", "4xCO2_cummins_ebm3.csv")
DEFAULT_VOLCANIC = os.path.join(HERE, "data", "volcanic_ERF_monthly_175001-201912.csv")


class SimpleFAIR:
    """Single-scenario, single-config FaIR wrapper with ECS control and GMST nudging.

    Parameters
    ----------
    scenario : str
        RCMIP scenario name (e.g. 'ssp245', 'ssp119').
    esm : str
        ESM config identifier from the calibration CSV, formatted as 'MODEL_RUN'
        (e.g. 'ACCESS-CM2_r1i1p1f1'). Use SimpleFAIR.list_esms() to see options.
    start_year : int
        First year of the simulation (default 1750). Use 1850 to skip the
        pre-industrial spinup and save ~30% compute time. Gas concentrations
        are initialized at pre-industrial baselines regardless.
    end_year : int
        Last year of the simulation.
    stochastic : bool
        Enable stochastic internal variability (Cummins et al. 2020).
    seed : int or None
        Random seed for reproducible stochastic runs.
    ebm_config_file : str
        Path to the EBM calibration CSV.
    volcanic_forcing_file : str or None
        Path to volcanic forcing CSV. None to skip volcanic forcing.
    """

    SPECIES = [
        # GHGs (emissions-driven)
        "CO2 FFI", "CO2 AFOLU", "CO2", "CH4", "N2O",
        # Aerosol precursors (emissions-driven, feed into ARI/ACI/ozone)
        "Sulfur", "BC", "OC", "NH3", "NOx", "VOC", "CO",
        # Halogens (emissions-driven, feed into EESC/ozone)
        "CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115",
        "HCFC-22", "HCFC-141b", "HCFC-142b",
        "CCl4", "CHCl3", "CH2Cl2", "CH3Cl", "CH3CCl3", "CH3Br",
        "Halon-1202", "Halon-1211", "Halon-1301", "Halon-2402",
        # F-gases (emissions-driven)
        "CF4", "C2F6", "C3F8", "c-C4F8", "C4F10", "C5F12", "C6F14",
        "C7F16", "C8F18", "NF3", "SF6", "SO2F2",
        "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", "HFC-227ea",
        "HFC-23", "HFC-236fa", "HFC-245fa", "HFC-32", "HFC-365mfc",
        "HFC-4310mee",
        # Aviation
        "NOx aviation",
        # Prescribed forcing
        "Solar", "Volcanic",
        # Calculated forcing agents
        "Aerosol-radiation interactions",
        "Aerosol-cloud interactions",
        "Ozone",
        "Contrails",
        "Light absorbing particles on snow and ice",
        "Stratospheric water vapour",
        "Land use",
        "Equivalent effective stratospheric chlorine",
    ]

    def __init__(
        self,
        scenario="ssp245",
        esm="GISS-E2-1-G_r1i1p3f1",
        start_year=1850,
        end_year=2100,
        stochastic=True,
        seed=42,
        ebm_config_file=DEFAULT_EBM_CONFIG,
        volcanic_forcing_file=DEFAULT_VOLCANIC,
    ):
        self._scenario = scenario
        self._esm = esm
        self._start_year = start_year
        self._end_year = end_year
        self._stochastic = stochastic
        self._seed = seed
        self._has_run = False
        self._prescribed_years = None
        self._prescribed_temperatures = None

        # Load EBM calibration data and find the ESM row
        self._ebm_df = pd.read_csv(ebm_config_file)
        self._ebm_df["config_name"] = (
            self._ebm_df["model"] + "_" + self._ebm_df["run"]
        )
        row = self._ebm_df[self._ebm_df["config_name"] == esm]
        if len(row) == 0:
            available = sorted(self._ebm_df["config_name"].tolist())
            raise ValueError(
                f"ESM '{esm}' not found. Available: {available}"
            )
        self._esm_row = row.iloc[0]

        # Build the internal FAIR object
        f = FAIR(ghg_method="meinshausen2020", ch4_method="thornhill2021")
        f.define_time(start_year, end_year + 1, 1)
        f.define_scenarios([scenario])
        f.define_configs([esm])

        # Species setup — use default properties (CO2 as 'calculated' from FFI+AFOLU)
        all_species, all_properties = read_properties()
        properties = {s: all_properties[s] for s in self.SPECIES}
        f.define_species(self.SPECIES, properties)

        f.allocate()
        f.fill_from_rcmip()

        # Volcanic forcing
        if volcanic_forcing_file is not None and os.path.exists(volcanic_forcing_file):
            df_volcanic = pd.read_csv(volcanic_forcing_file, index_col="year")
            n_timebounds = len(f.timebounds)
            volcanic_forcing = np.zeros(n_timebounds)
            annual_volcanic = (
                df_volcanic.loc[1749:]
                .groupby(np.ceil(df_volcanic.loc[1749:].index) // 1)
                .mean()
                .squeeze()
                .values
            )
            n_fill = min(len(annual_volcanic), n_timebounds)
            volcanic_forcing[:n_fill] = annual_volcanic[:n_fill]
            fill(f.forcing, volcanic_forcing[:, None, None], specie="Volcanic")

        # Species configs
        f.fill_species_configs()

        # Initialize state
        initialise(f.concentration, f.species_configs["baseline_concentration"])
        initialise(f.forcing, 0)
        initialise(f.temperature, 0)
        initialise(f.cumulative_emissions, 0)
        initialise(f.airborne_emissions, 0)

        # Fill climate configs for this single ESM
        r = self._esm_row
        fill(
            f.climate_configs["ocean_heat_capacity"],
            np.array([r["C1"], r["C2"], r["C3"]]),
            config=esm,
        )
        fill(
            f.climate_configs["ocean_heat_transfer"],
            np.array([r["kappa1"], r["kappa2"], r["kappa3"]]),
            config=esm,
        )
        fill(f.climate_configs["deep_ocean_efficacy"], r["epsilon"], config=esm)
        fill(f.climate_configs["gamma_autocorrelation"], r["gamma"], config=esm)
        fill(f.climate_configs["sigma_eta"], r["sigma_eta"], config=esm)
        fill(f.climate_configs["sigma_xi"], r["sigma_xi"], config=esm)
        fill(f.climate_configs["forcing_4co2"], r["F_4xCO2"], config=esm)
        fill(f.climate_configs["stochastic_run"], stochastic, config=esm)
        fill(f.climate_configs["use_seed"], seed is not None, config=esm)
        if seed is not None:
            fill(f.climate_configs["seed"], seed, config=esm)

        self._fair = f

    def copy(self, seed=None):
        """Create a lightweight clone ready to run, without re-parsing RCMIP data.

        Returns a new SimpleFAIR instance that shares no mutable state with
        the original. ~1ms vs ~1s for a fresh constructor call.

        Parameters
        ----------
        seed : int or None
            New random seed. If None, uses the original seed.
        """
        if self._has_run:
            raise RuntimeError(
                "Cannot copy after run() — the internal state has been "
                "modified. Copy from a fresh (un-run) instance."
            )
        clone = object.__new__(SimpleFAIR)
        clone._scenario = self._scenario
        clone._esm = self._esm
        clone._start_year = self._start_year
        clone._end_year = self._end_year
        clone._stochastic = self._stochastic
        clone._seed = seed if seed is not None else self._seed
        clone._has_run = False
        clone._prescribed_years = None
        clone._prescribed_temperatures = None
        clone._ebm_df = self._ebm_df
        clone._esm_row = self._esm_row
        clone._fair = copy.deepcopy(self._fair)

        # Update seed in the cloned FAIR object
        if seed is not None:
            fill(clone._fair.climate_configs["use_seed"], True, config=self._esm)
            fill(clone._fair.climate_configs["seed"], seed, config=self._esm)

        return clone

    def set_ecs(self, ecs_desired):
        """Set equilibrium climate sensitivity by adjusting the feedback parameter.

        Must be called before run(). ECS = F_2xCO2 / kappa1, so
        kappa1_new = F_2xCO2 / ECS_desired.

        Parameters
        ----------
        ecs_desired : float
            Target ECS in Kelvin.
        """
        if self._has_run:
            raise RuntimeError("Cannot set ECS after run(). Create a new instance.")
        forcing_4co2 = self._fair.climate_configs["forcing_4co2"].values[0]
        kappa1_new = (forcing_4co2 * 0.5) / ecs_desired
        fill(
            self._fair.climate_configs["ocean_heat_transfer"],
            kappa1_new,
            config=self._esm,
            layer=0,
        )

    def prescribe_gmst(self, years, temperatures):
        """Prescribe observed GMST anomaly for nudging.

        During the prescribed period, the full gas cycle and forcing pipeline runs
        normally, but surface temperature is overwritten with the prescribed values
        after each EBM step. Deep ocean layers evolve consistently. At the end of
        the prescribed period, the model transitions to free-running mode.

        Parameters
        ----------
        years : array-like
            Calendar years for prescribed temperatures (must be contiguous integers).
        temperatures : array-like
            GMST anomaly relative to pre-industrial (K), same length as years.
        """
        if self._has_run:
            raise RuntimeError(
                "Cannot prescribe GMST after run(). Create a new instance."
            )
        years = np.asarray(years, dtype=int)
        temperatures = np.asarray(temperatures, dtype=float)
        if len(years) != len(temperatures):
            raise ValueError("years and temperatures must have the same length.")
        if years[0] < self._start_year or years[-1] > self._end_year:
            raise ValueError(
                f"Prescribed years must be within [{self._start_year}, {self._end_year}]."
            )
        self._prescribed_years = years
        self._prescribed_temperatures = temperatures

    def run(self):
        """Run the model.

        Replicates FAIR's run loop with an additional nudging step: if GMST has been
        prescribed via prescribe_gmst(), surface temperature is overwritten during
        the prescribed period while deep ocean layers evolve naturally.
        """
        if self._has_run:
            raise RuntimeError("Already run. Create a new instance for a new run.")

        f = self._fair
        f._check_properties()
        f._make_indices()

        # Build EBMs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f._make_ebms()

        # CO2 summation (if CO2 is in 'calculated' mode from FFI+AFOLU)
        if f._routine_flags.get("ghg", False):
            if hasattr(f, "_co2_indices") and hasattr(f, "_co2_ffi_indices"):
                if np.any(f._co2_ffi_indices) and np.any(f._co2_indices):
                    f.emissions.data[..., f._co2_indices] = (
                        np.nansum(
                            f.emissions.data[
                                ...,
                                f._co2_ffi_indices | f._co2_afolu_indices,
                            ],
                            axis=SPECIES_AXIS,
                            keepdims=True,
                        )
                    )

        # Cumulative emissions
        f.cumulative_emissions[1:, ...] = (
            f.emissions.cumsum(dim="timepoints", skipna=False) * f.timestep
            + f.cumulative_emissions[0, ...]
        ).data

        # Extract numpy arrays from xarray
        alpha_lifetime_array = f.alpha_lifetime.data
        airborne_emissions_array = f.airborne_emissions.data
        baseline_concentration_array = f.species_configs["baseline_concentration"].data
        baseline_emissions_array = f.species_configs["baseline_emissions"].data
        br_atoms_array = f.species_configs["br_atoms"].data
        ch4_lifetime_chemical_sensitivity_array = f.species_configs[
            "ch4_lifetime_chemical_sensitivity"
        ].data
        lifetime_temperature_sensitivity_array = f.species_configs[
            "lifetime_temperature_sensitivity"
        ].data
        cl_atoms_array = f.species_configs["cl_atoms"].data
        concentration_array = f.concentration.data
        concentration_per_emission_array = f.species_configs[
            "concentration_per_emission"
        ].data
        contrails_radiative_efficiency_array = f.species_configs[
            "contrails_radiative_efficiency"
        ].data
        cummins_state_array = (
            np.ones(
                (f._n_timebounds, f._n_scenarios, f._n_configs, f._n_layers + 1)
            )
            * np.nan
        )
        cumulative_emissions_array = f.cumulative_emissions.data
        deep_ocean_efficacy_array = f.climate_configs["deep_ocean_efficacy"].data
        emissions_array = f.emissions.data
        erfari_radiative_efficiency_array = f.species_configs[
            "erfari_radiative_efficiency"
        ].data
        erfaci_scale_array = f.species_configs["aci_scale"].data
        erfaci_shape_array = f.species_configs["aci_shape"].data
        forcing_array = f.forcing.data
        forcing_scale_array = f.species_configs["forcing_scale"].data * (
            1 + f.species_configs["tropospheric_adjustment"].data
        )
        forcing_efficacy_array = f.species_configs["forcing_efficacy"].data
        forcing_efficacy_sum_array = (
            np.ones((f._n_timebounds, f._n_scenarios, f._n_configs)) * np.nan
        )
        forcing_reference_concentration_array = f.species_configs[
            "forcing_reference_concentration"
        ].data
        forcing_sum_array = f.forcing_sum.data
        forcing_temperature_feedback_array = f.species_configs[
            "forcing_temperature_feedback"
        ].data
        fractional_release_array = f.species_configs["fractional_release"].data
        g0_array = f.species_configs["g0"].data
        g1_array = f.species_configs["g1"].data
        gas_partitions_array = f.gas_partitions.data
        greenhouse_gas_radiative_efficiency_array = f.species_configs[
            "greenhouse_gas_radiative_efficiency"
        ].data
        h2o_stratospheric_factor_array = f.species_configs[
            "h2o_stratospheric_factor"
        ].data
        iirf_0_array = f.species_configs["iirf_0"].data
        iirf_airborne_array = f.species_configs["iirf_airborne"].data
        iirf_temperature_array = f.species_configs["iirf_temperature"].data
        iirf_uptake_array = f.species_configs["iirf_uptake"].data
        land_use_cumulative_emissions_to_forcing_array = f.species_configs[
            "land_use_cumulative_emissions_to_forcing"
        ].data
        lapsi_radiative_efficiency_array = f.species_configs[
            "lapsi_radiative_efficiency"
        ].data
        ocean_heat_transfer_array = f.climate_configs["ocean_heat_transfer"].data
        ozone_radiative_efficiency_array = f.species_configs[
            "ozone_radiative_efficiency"
        ].data
        partition_fraction_array = f.species_configs["partition_fraction"].data
        unperturbed_lifetime_array = f.species_configs["unperturbed_lifetime"].data

        if f._routine_flags["temperature"]:
            eb_matrix_d_array = f.ebms["eb_matrix_d"].data
            forcing_vector_d_array = f.ebms["forcing_vector_d"].data
            stochastic_d_array = f.ebms["stochastic_d"].data

        # Initial forcing sum
        forcing_sum_array[0:1, ...] = np.nansum(
            forcing_array[0:1, ...], axis=SPECIES_AXIS
        )

        # Initialize state vector: [eta, T_surface, T_deep1, T_deep2]
        cummins_state_array[0, ..., 0] = forcing_sum_array[0, ...]
        cummins_state_array[..., 1:] = f.temperature.data

        # Meinshausen2020 forcing offset
        if f._routine_flags["ghg"]:
            if not hasattr(f, "ghg_forcing_offset"):
                f.ghg_forcing_offset = meinshausen2020(
                    baseline_concentration_array[None, None, ...],
                    forcing_reference_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ...],
                    greenhouse_gas_radiative_efficiency_array[None, None, ...],
                    f._co2_indices,
                    f._ch4_indices,
                    f._n2o_indices,
                    f._minor_ghg_indices,
                )

        # Nudging setup
        nudge_start_idx = None
        nudge_end_idx = None
        if self._prescribed_temperatures is not None:
            nudge_start_idx = int(self._prescribed_years[0] - self._start_year)
            nudge_end_idx = int(self._prescribed_years[-1] - self._start_year)

        # ===== MAIN LOOP =====
        for i_timepoint in range(f._n_timepoints):
            if f._routine_flags["ghg"]:
                # 1. Alpha scaling
                alpha_lifetime_array[
                    i_timepoint : i_timepoint + 1, ..., f._ghg_indices
                ] = calculate_alpha(
                    airborne_emissions_array[
                        i_timepoint : i_timepoint + 1, ..., f._ghg_indices
                    ],
                    cumulative_emissions_array[
                        i_timepoint : i_timepoint + 1, ..., f._ghg_indices
                    ],
                    g0_array[None, None, ..., f._ghg_indices],
                    g1_array[None, None, ..., f._ghg_indices],
                    iirf_0_array[None, None, ..., f._ghg_indices],
                    iirf_airborne_array[None, None, ..., f._ghg_indices],
                    iirf_temperature_array[None, None, ..., f._ghg_indices],
                    iirf_uptake_array[None, None, ..., f._ghg_indices],
                    cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2],
                    f.iirf_max,
                )

                # 2. CH4 lifetime (thornhill2021)
                if f.ch4_method == "thornhill2021":
                    alpha_lifetime_array[
                        i_timepoint : i_timepoint + 1, ..., f._ch4_indices
                    ] = calculate_alpha_ch4(
                        emissions_array[i_timepoint : i_timepoint + 1, ...],
                        concentration_array[i_timepoint : i_timepoint + 1, ...],
                        cummins_state_array[
                            i_timepoint : i_timepoint + 1, ..., 1:2
                        ],
                        baseline_emissions_array[None, None, ...],
                        baseline_concentration_array[None, None, ...],
                        ch4_lifetime_chemical_sensitivity_array[None, None, ...],
                        lifetime_temperature_sensitivity_array[
                            None, None, :, None
                        ],
                        f._aerosol_chemistry_from_emissions_indices,
                        f._aerosol_chemistry_from_concentration_indices,
                    )

                # 3. Emissions to concentrations (forward)
                (
                    concentration_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        f._ghg_forward_indices,
                    ],
                    gas_partitions_array[..., f._ghg_forward_indices, :],
                    airborne_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        f._ghg_forward_indices,
                    ],
                ) = step_concentration(
                    emissions_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        f._ghg_forward_indices,
                        None,
                    ],
                    gas_partitions_array[..., f._ghg_forward_indices, :],
                    airborne_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2,
                        ...,
                        f._ghg_forward_indices,
                        None,
                    ],
                    alpha_lifetime_array[
                        i_timepoint : i_timepoint + 1,
                        ...,
                        f._ghg_forward_indices,
                        None,
                    ],
                    baseline_concentration_array[
                        None, None, ..., f._ghg_forward_indices
                    ],
                    baseline_emissions_array[
                        None, None, ..., f._ghg_forward_indices, None
                    ],
                    concentration_per_emission_array[
                        None, None, ..., f._ghg_forward_indices
                    ],
                    unperturbed_lifetime_array[
                        None, None, ..., f._ghg_forward_indices, :
                    ],
                    partition_fraction_array[
                        None, None, ..., f._ghg_forward_indices, :
                    ],
                    f.timestep,
                )

                # 4. Concentrations to emissions (inverse)
                if np.any(f._ghg_inverse_indices):
                    (
                        emissions_array[
                            i_timepoint : i_timepoint + 1,
                            ...,
                            f._ghg_inverse_indices,
                        ],
                        gas_partitions_array[..., f._ghg_inverse_indices, :],
                        airborne_emissions_array[
                            i_timepoint + 1 : i_timepoint + 2,
                            ...,
                            f._ghg_inverse_indices,
                        ],
                    ) = unstep_concentration(
                        concentration_array[
                            i_timepoint + 1 : i_timepoint + 2,
                            ...,
                            f._ghg_inverse_indices,
                        ],
                        gas_partitions_array[
                            None, ..., f._ghg_inverse_indices, :
                        ],
                        airborne_emissions_array[
                            i_timepoint : i_timepoint + 1,
                            ...,
                            f._ghg_inverse_indices,
                            None,
                        ],
                        alpha_lifetime_array[
                            i_timepoint : i_timepoint + 1,
                            ...,
                            f._ghg_inverse_indices,
                            None,
                        ],
                        baseline_concentration_array[
                            None, None, ..., f._ghg_inverse_indices
                        ],
                        baseline_emissions_array[
                            None, None, ..., f._ghg_inverse_indices
                        ],
                        concentration_per_emission_array[
                            None, None, ..., f._ghg_inverse_indices
                        ],
                        unperturbed_lifetime_array[
                            None, None, ..., f._ghg_inverse_indices, :
                        ],
                        partition_fraction_array[
                            None, None, ..., f._ghg_inverse_indices, :
                        ],
                        f.timestep,
                    )
                    cumulative_emissions_array[
                        i_timepoint + 1, ..., f._ghg_inverse_indices
                    ] = (
                        cumulative_emissions_array[
                            i_timepoint, ..., f._ghg_inverse_indices
                        ]
                        + emissions_array[
                            i_timepoint, ..., f._ghg_inverse_indices
                        ]
                        * f.timestep
                    )

                # 5. GHG forcing (meinshausen2020)
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._ghg_indices
                ] = meinshausen2020(
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    forcing_reference_concentration_array[None, None, ...]
                    * np.ones(
                        (1, f._n_scenarios, f._n_configs, f._n_species)
                    ),
                    forcing_scale_array[None, None, ...],
                    greenhouse_gas_radiative_efficiency_array[None, None, ...],
                    f._co2_indices,
                    f._ch4_indices,
                    f._n2o_indices,
                    f._minor_ghg_indices,
                )[
                    0:1, ..., f._ghg_indices
                ]
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._ghg_indices
                ] = (
                    forcing_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., f._ghg_indices
                    ]
                    - f.ghg_forcing_offset[..., f._ghg_indices]
                )

            # 6. Aerosol direct forcing
            if f._routine_flags["ari"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._ari_indices
                ] = calculate_erfari_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_emissions_array[None, None, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ..., f._ari_indices],
                    erfari_radiative_efficiency_array[None, None, ...],
                    f._aerosol_chemistry_from_emissions_indices,
                    f._aerosol_chemistry_from_concentration_indices,
                )

            # 7. Aerosol indirect forcing
            if f._routine_flags["aci"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._aci_indices
                ] = logsum(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_emissions_array[None, None, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ..., f._aci_indices],
                    erfaci_scale_array[None, None, :],
                    erfaci_shape_array[None, None, ...],
                    f._aerosol_chemistry_from_emissions_indices,
                    f._aerosol_chemistry_from_concentration_indices,
                )

            # 8. EESC
            if f._routine_flags["eesc"]:
                concentration_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._eesc_indices
                ] = calculate_eesc(
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    fractional_release_array[None, None, ...],
                    cl_atoms_array[None, None, ...],
                    br_atoms_array[None, None, ...],
                    f._cfc11_indices,
                    f._halogen_indices,
                    f.br_cl_ods_potential,
                )

            # 9. Ozone forcing
            if f._routine_flags["ozone"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._ozone_indices
                ] = thornhill2021(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_emissions_array[None, None, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ..., f._ozone_indices],
                    ozone_radiative_efficiency_array[None, None, ...],
                    f._aerosol_chemistry_from_emissions_indices,
                    f._aerosol_chemistry_from_concentration_indices,
                )

            # 10. Contrails
            if f._routine_flags["contrails"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._contrails_indices
                ] = calculate_linear_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    0,
                    forcing_scale_array[None, None, ..., f._contrails_indices],
                    contrails_radiative_efficiency_array[None, None, ...],
                )

            # 11. LAPSI
            if f._routine_flags["lapsi"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._lapsi_indices
                ] = calculate_linear_forcing(
                    emissions_array[i_timepoint : i_timepoint + 1, ...],
                    baseline_emissions_array[None, None, ...],
                    forcing_scale_array[None, None, ..., f._lapsi_indices],
                    lapsi_radiative_efficiency_array[None, None, ...],
                )

            # 12. Stratospheric H2O
            if f._routine_flags["h2o stratospheric"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._h2ostrat_indices
                ] = calculate_linear_forcing(
                    concentration_array[i_timepoint + 1 : i_timepoint + 2, ...],
                    baseline_concentration_array[None, None, ...],
                    forcing_scale_array[None, None, ..., f._h2ostrat_indices],
                    h2o_stratospheric_factor_array[None, None, ...],
                )

            # 13. Land use
            if f._routine_flags["land use"]:
                forcing_array[
                    i_timepoint + 1 : i_timepoint + 2, ..., f._landuse_indices
                ] = calculate_linear_forcing(
                    cumulative_emissions_array[
                        i_timepoint + 1 : i_timepoint + 2, ...
                    ],
                    0,
                    forcing_scale_array[None, None, ..., f._landuse_indices],
                    land_use_cumulative_emissions_to_forcing_array[None, None, ...],
                )

            # 14. Temperature-forcing feedback
            forcing_array[i_timepoint + 1 : i_timepoint + 2, ...] = (
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
                + cummins_state_array[i_timepoint : i_timepoint + 1, ..., 1:2]
                * forcing_temperature_feedback_array[None, None, ...]
            )

            # 15. Sum forcings
            forcing_sum_array[i_timepoint + 1 : i_timepoint + 2, ...] = np.nansum(
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...],
                axis=SPECIES_AXIS,
            )
            forcing_efficacy_sum_array[
                i_timepoint + 1 : i_timepoint + 2, ...
            ] = np.nansum(
                forcing_array[i_timepoint + 1 : i_timepoint + 2, ...]
                * forcing_efficacy_array[None, None, ...],
                axis=SPECIES_AXIS,
            )

            # 16. Temperature step
            if f._routine_flags["temperature"]:
                cummins_state_array[
                    i_timepoint + 1 : i_timepoint + 2, ...
                ] = step_temperature(
                    cummins_state_array[i_timepoint : i_timepoint + 1, ...],
                    eb_matrix_d_array[None, None, ...],
                    forcing_vector_d_array[None, None, ...],
                    stochastic_d_array[
                        i_timepoint + 1 : i_timepoint + 2, None, ...
                    ],
                    forcing_efficacy_sum_array[
                        i_timepoint + 1 : i_timepoint + 2, ..., None
                    ],
                )

                # NUDGING: overwrite surface T during prescribed period
                if nudge_start_idx is not None:
                    tb_idx = i_timepoint + 1  # timebound index
                    if nudge_start_idx <= tb_idx <= nudge_end_idx:
                        prescribed_idx = tb_idx - nudge_start_idx
                        cummins_state_array[tb_idx, :, :, 1] = (
                            self._prescribed_temperatures[prescribed_idx]
                        )

        # 17. TOA imbalance
        toa_imbalance_array = calculate_toa_imbalance_postrun(
            cummins_state_array,
            forcing_sum_array,
            ocean_heat_transfer_array,
            deep_ocean_efficacy_array,
        )

        # 18. Ocean heat content change
        ocean_heat_content_change_array = f.ocean_heat_content_change[0:1, ...] + (
            (
                np.cumsum(toa_imbalance_array, axis=TIME_AXIS)
                - toa_imbalance_array[0:1, ...]
            )
            * f.timestep
            * earth_radius**2
            * 4
            * np.pi
            * seconds_per_year
        )

        # 19. Airborne fraction
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            airborne_fraction_array = (
                airborne_emissions_array / cumulative_emissions_array
            )

        # 20. Write back to xarray
        f.temperature.data = cummins_state_array[..., 1:]
        f.concentration.data = concentration_array
        f.emissions.data = emissions_array
        f.forcing.data = forcing_array
        f.forcing_sum.data = forcing_sum_array
        f.cumulative_emissions.data = cumulative_emissions_array
        f.airborne_emissions.data = airborne_emissions_array
        f.airborne_fraction.data = airborne_fraction_array
        f.gas_partitions.data = gas_partitions_array
        f.ocean_heat_content_change.data = ocean_heat_content_change_array
        f.toa_imbalance.data = toa_imbalance_array
        f.stochastic_forcing.data = cummins_state_array[..., 0]

        self._has_run = True

    def get_gmst(self):
        """Return GMST anomaly (surface layer temperature).

        Returns
        -------
        timebounds : np.ndarray
            Calendar years (timebounds).
        gmst : np.ndarray
            Global mean surface temperature anomaly (K).
        """
        if not self._has_run:
            raise RuntimeError("Must call run() first.")
        f = self._fair
        timebounds = f.timebounds
        gmst = f.temperature.sel(layer=0).values.squeeze()
        return timebounds, gmst

    def get_forcing(self):
        """Return total radiative forcing.

        Returns
        -------
        timebounds : np.ndarray
            Calendar years.
        forcing : np.ndarray
            Total radiative forcing (W/m2).
        """
        if not self._has_run:
            raise RuntimeError("Must call run() first.")
        return self._fair.timebounds, self._fair.forcing_sum.values.squeeze()

    def get_concentration(self, specie="CO2"):
        """Return atmospheric concentration for a species.

        Parameters
        ----------
        specie : str
            Species name.

        Returns
        -------
        timebounds : np.ndarray
            Calendar years.
        concentration : np.ndarray
            Atmospheric concentration (ppm for CO2, ppb for CH4/N2O).
        """
        if not self._has_run:
            raise RuntimeError("Must call run() first.")
        return (
            self._fair.timebounds,
            self._fair.concentration.sel(specie=specie).values.squeeze(),
        )

    @property
    def ecs(self):
        """Equilibrium climate sensitivity (K) from the EBM emergent parameters."""
        if not self._has_run:
            raise RuntimeError("Must call run() first (EBMs built during run).")
        return float(self._fair.ebms["ecs"].values[0])

    @staticmethod
    def list_esms(ebm_config_file=DEFAULT_EBM_CONFIG):
        """List available ESM config identifiers.

        Returns
        -------
        list of str
            Available 'MODEL_RUN' identifiers.
        """
        df = pd.read_csv(ebm_config_file)
        return sorted((df["model"] + "_" + df["run"]).tolist())
