.PHONY: reproduce simulation timing test clean

PYTHON ?= python3

SIM_TABLE := results/tables/simulation_summary.csv
TIME_TABLE := results/tables/timing.csv
TIME_FIGURE := results/figures/timing.png

CORE_SOURCES := src/CLE.py src/CLS.py src/generation.py

# From a clean clone, these two real targets generate all designated outputs.
reproduce: $(SIM_TABLE) $(TIME_TABLE)

simulation: $(SIM_TABLE)

$(SIM_TABLE): src/simulation.py $(CORE_SOURCES)
	@mkdir -p results/tables
	$(PYTHON) -m src.simulation --n 500 --replications 100 --seed 123 --network PowerLaw --output $(SIM_TABLE)

timing: $(TIME_TABLE)

# timing.py writes both the CSV target and the timing figure in one run.
$(TIME_TABLE): src/timing.py $(CORE_SOURCES)
	@mkdir -p results/tables results/figures
	$(PYTHON) -m src.timing --repeats 5 --seed 123 --network Dyad --table $(TIME_TABLE) --figure $(TIME_FIGURE)

test:
	$(PYTHON) -m pytest -q

clean:
	rm -f $(SIM_TABLE) $(TIME_TABLE) $(TIME_FIGURE)
