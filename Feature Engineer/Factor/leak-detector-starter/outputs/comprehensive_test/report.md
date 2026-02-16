# Leak Detection Summary

发现如下可疑因子：

| factor | signal_type | confidence |
| --- | --- | --- |
| explicit_leak_future_price | forward_replay | 1.000 |
| explicit_leak_future_price | consistency_check | 1.000 |
| explicit_leak_future_return | forward_replay | 1.000 |
| explicit_leak_future_return | consistency_check | 1.000 |
| safe_close_ma5 | forward_replay | 0.022 |
| safe_momentum_5d | forward_replay | 0.027 |
| safe_volatility | forward_replay | 0.024 |
| safe_volume_growth | forward_replay | 0.023 |
