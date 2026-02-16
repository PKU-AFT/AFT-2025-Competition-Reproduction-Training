# Leak Detection Summary

发现如下可疑因子：

| factor | signal_type | confidence |
| --- | --- | --- |
| explicit_leak_future_close_2 | forward_replay | 1.000 |
| explicit_leak_future_close_2 | consistency_check | 1.000 |
| explicit_leak_future_high_5 | forward_replay | 1.000 |
| explicit_leak_future_high_5 | consistency_check | 1.000 |
| explicit_leak_future_return_1 | forward_replay | 1.000 |
| explicit_leak_future_return_1 | consistency_check | 1.000 |
| implicit_leak_close_ma_ratio | forward_replay | 0.002 |
| implicit_leak_current_volume_norm | forward_replay | 0.002 |
| safe_close_ma5 | forward_replay | 0.002 |
| safe_momentum_5 | forward_replay | 0.002 |
| safe_range_ratio | forward_replay | 0.002 |
| safe_volatility_10 | forward_replay | 0.002 |
| safe_volume_ma10 | forward_replay | 0.002 |
| safe_vwap_deviation | forward_replay | 0.002 |
