# Data

`Asylum_data.csv` (39 200 rows × 69 columns) is a dyadic panel: 200 origin countries × 28 European
destinations × 7 years (2010–2016). It was provided for the course and is **not redistributed here**.
Place the file in this folder to run `analysis/run_analysis.py` or `R/gravity_fixest.R`.

Variables used:

| Variable | Meaning |
|---|---|
| `number_app_tot` | asylum applications from origin *j* to destination *k* in year *t* (82 % zeros) |
| `diasp_tot_10` | stock of origin-born migrants in the destination in 2010 |
| `distcap` | distance between capitals (km) |
| `comlang_off`, `colony`, `contig` | common official language, colonial link, common border |
| `gdpcap_d`, `hdi_d` | destination GDP per capita (missing in 2016) and HDI (missing in 2015–16) |
| `fh_civlib_o`, `fh_polrights_o` | Freedom House civil liberties / political rights, 1 (free) to 7 (not free) |
| `fh_polity2_o` | Freedom House / Polity composite, 0 to 10, **10 = most democratic** |
| `ptscale_o` | Political Terror Scale, 1 to 5 |
