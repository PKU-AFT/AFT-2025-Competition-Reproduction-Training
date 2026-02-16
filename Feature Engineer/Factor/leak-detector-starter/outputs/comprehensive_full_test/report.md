# Leak Detection Summary

发现如下可疑因子：

| factor | signal_type | confidence |
| --- | --- | --- |
| explicit_leak_future_price | forward_replay | 1.000 |
| explicit_leak_future_price | consistency_check | 1.000 |
| explicit_leak_future_return | forward_replay | 1.000 |
| explicit_leak_future_return | consistency_check | 1.000 |
| implicit_leak_close_momentum | forward_replay | 0.027 |
| implicit_leak_daily_range | forward_replay | 0.022 |
| implicit_leak_high_ratio | forward_replay | 0.022 |
| implicit_leak_volume_norm | forward_replay | 0.022 |
| safe_open_momentum | forward_replay | 0.027 |
| safe_price_ma5 | forward_replay | 0.022 |
| safe_volume_growth | forward_replay | 0.023 |
