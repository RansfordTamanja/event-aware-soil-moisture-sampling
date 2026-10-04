| comparison               | metric   | days                |   n_days |   rmse_event_aware |   rmse_other |   mse_ratio |   event_aware_better_days |   share_of_total_reduction |
|:-------------------------|:---------|:--------------------|---------:|-------------------:|-------------:|------------:|--------------------------:|---------------------------:|
| Event-aware vs Fixed-120 | interp   | event-affected days |      104 |            0.00066 |      0.00271 |     0.06006 |                        50 |                    1.00377 |
| Event-aware vs Fixed-120 | interp   | quiet days          |       58 |            0.00036 |      0.00029 |     1.56602 |                         0 |                   -0.00377 |
| Event-aware vs Fixed-120 | hold     | event-affected days |      104 |            0.00276 |      0.00529 |     0.27189 |                        46 |                    1.02691 |
| Event-aware vs Fixed-120 | hold     | quiet days          |       58 |            0.00116 |      0.00061 |     3.53475 |                         0 |                   -0.02691 |
| Event-aware vs Reactive  | interp   | event-affected days |      104 |            0.00066 |      0.00396 |     0.02819 |                        70 |                    1.00021 |
| Event-aware vs Reactive  | interp   | quiet days          |       58 |            0.00036 |      0.00035 |     1.04691 |                        29 |                   -0.00021 |
| Event-aware vs Reactive  | hold     | event-affected days |      104 |            0.00276 |      0.00678 |     0.16549 |                        58 |                    1.00981 |
| Event-aware vs Reactive  | hold     | quiet days          |       58 |            0.00116 |      0.00082 |     2.00273 |                        25 |                   -0.00981 |
| Event-aware vs Fixed     | interp   | event-affected days |      104 |            0.00066 |      0.00445 |     0.02226 |                        76 |                    0.99992 |
| Event-aware vs Fixed     | interp   | quiet days          |       58 |            0.00036 |      0.00036 |     0.97856 |                        21 |                    0.00008 |
| Event-aware vs Fixed     | hold     | event-affected days |      104 |            0.00276 |      0.00897 |     0.09452 |                        77 |                    1.00057 |
| Event-aware vs Fixed     | hold     | quiet days          |       58 |            0.00116 |      0.00112 |     1.05873 |                        15 |                   -0.00057 |