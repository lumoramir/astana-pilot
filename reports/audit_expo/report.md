# Road-safety audit: astana_expo

Probe driver: **expert**. Events are surrogate safety measures (near-misses), not crashes.

## What-if comparison (identical trips)

| variant | trips | km | success | collisions | mean trip (min) | weighted events / 100 km |
|---|---|---|---|---|---|---|
| baseline | 96 | 36.4 | 75% | 17 | 1.6 | 3311 [2842..3847] |

## Events per 100 km

| variant | collision | aeb | ttc | red_light | yellow_light | stop_sign | yield | zebra | ped_close | hard_brake | phantom_brake |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 46.7 | 307.6 | 255.4 | 0.0 | 0.0 | 0.0 | 0.0 | 2.7 | 5.5 | 1153.5 | 324.1 |

## Top hot-spots (baseline)

| # | street | location | events/km | AEB | TTC<1.5s | violations | ped. close | lat, lon |
|---|---|---|---|---|---|---|---|---|
| 1 | Alikhan Bokeikhan Street | junction 8087530293 (priority) | 367.4 | 3 | 2 | 0 | 0 | 51.09664, 71.43101 |
| 2 | 28/5 көше көшесі | junction 11140220779 (priority) | 199.1 | 0 | 1 | 0 | 0 | 51.09507, 71.41382 |
| 3 | Хейдар Әлиев көшесі | junction 4985151137 (priority) | 186.7 | 1 | 1 | 0 | 0 | 51.09552, 71.41060 |
| 4 | Жасыл Саябақ-223 тұйық көшесі | junction 2505673175 (priority) | 150.4 | 1 | 2 | 0 | 0 | 51.09035, 71.43101 |
| 5 | Mangilik El Avenue | junction 4463429076 (priority) | 133.2 | 2 | 1 | 0 | 0 | 51.09867, 71.42635 |
| 6 | Ақмешіт көшесі | junction 4520076207 (priority) | 106.8 | 1 | 0 | 0 | 0 | 51.08877, 71.41168 |
| 7 | Alikhan Bokeikhan Street | junction 5281680461 (priority) | 104.3 | 0 | 0 | 0 | 0 | 51.09934, 71.43172 |
| 8 | 28/5 көше көшесі | junction 9879179071 (priority) | 101.6 | 1 | 1 | 0 | 0 | 51.09350, 71.41347 |
| 9 | Хейдар Әлиев көшесі | junction 4985151161 (signalised) | 100.3 | 1 | 1 | 0 | 0 | 51.09372, 71.40989 |
| 10 | Uly Dala Avenue | junction 4382872222 (signalised) | 79.4 | 1 | 1 | 0 | 0 | 51.09507, 71.43029 |
| 11 | Mangilik El Avenue | junction 8084055198 (priority) | 77.9 | 0 | 1 | 0 | 0 | 51.09732, 71.42599 |
| 12 | Хейдар Әлиев көшесі | junction 7209492046 (priority) | 77.8 | 1 | 1 | 0 | 0 | 51.09237, 71.40917 |

*Method:* weights collision 10, zebra / red-light / stop violations 4-5, AEB 3, pedestrian < 1.2 m 3, TTC < 1.5 s 2, hard braking 1, needless crawling 0.7; rate = weighted events per km driven through a 25 m cell (cells with < 300 m of exposure ignored).