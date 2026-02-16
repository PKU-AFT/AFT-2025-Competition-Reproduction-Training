"""Test script demonstrating time flattening integration in leak detection framework.

This script creates sample data with discrete time columns and runs leak detection
with automatic time flattening enabled.
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from leak_detection.pipeline import run_pipeline
from leak_detection.time_utils import TimeIndexConfig


def create_test_data() -> Path:
    """Create sample factor data with discrete time columns."""

    # Simulate factor data with separate year/month/day columns
    data = []
    for year in [2023, 2024]:
        for month in [1, 2]:
            for day in [15, 20, 25]:
                for symbol in ['AAPL', 'GOOGL', 'MSFT']:
                    data.append({
                        'year': year,
                        'month': month,
                        'day': day,
                        'symbol': symbol,
                        'open': 100 + (year - 2023) * 10 + month * 5 + day * 0.1,
                        'close': 102 + (year - 2023) * 10 + month * 5 + day * 0.1,
                        'volume': 1000000 + day * 10000,
                    })

    df = pd.DataFrame(data)

    # Save to CSV
    output_path = PROJECT_ROOT / 'data' / 'test_time_flatten.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Created test data: {output_path}")
    print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")
    print("\n前5行:")
    print(df.head())

    return output_path


def test_case_1_auto_flatten():
    """Test Case 1: Automatic time flattening with default config."""

    print("\n" + "=" * 70)
    print("测试案例 1: 自动时间展平（默认配置）")
    print("=" * 70)

    data_path = create_test_data()

    # Run with default time config (auto flattening enabled)
    artefacts = run_pipeline(
        data_path=data_path,
        output_dir='outputs/test_auto_flatten',
    )

    print("\n✅ 测试完成！生成的文件:")
    for name, path in artefacts.items():
        print(f"   {name}: {path}")

    # Read and display time report
    if 'time_report' in artefacts:
        print("\n📊 时间处理报告:")
        with open(artefacts['time_report'], 'r', encoding='utf-8') as f:
            print(f.read())

    return artefacts


def test_case_2_custom_config():
    """Test Case 2: Custom time configuration with verbose output."""

    print("\n" + "=" * 70)
    print("测试案例 2: 自定义时间配置（详细输出）")
    print("=" * 70)

    data_path = create_test_data()

    # Custom time config
    time_config = TimeIndexConfig(
        verbose=True,
        tz='Asia/Shanghai',
        index_name='trade_date',
        drop_originals=False,  # Keep original columns
    )

    artefacts = run_pipeline(
        data_path=data_path,
        output_dir='outputs/test_custom_config',
        time_config=time_config,
    )

    print("\n✅ 测试完成！")
    return artefacts


def test_case_3_non_standard_columns():
    """Test Case 3: Non-standard column names with manual mapping."""

    print("\n" + "=" * 70)
    print("测试案例 3: 非标准列名（手动映射）")
    print("=" * 70)

    # Create data with non-standard column names
    data = []
    for yr in [2023, 2024]:
        for mth in [1, 2]:
            for dy in [15, 25]:
                for sym in ['AAPL', 'GOOGL']:
                    data.append({
                        'fiscal_year': yr,
                        'report_month': mth,
                        'business_day': dy,
                        'stock_code': sym,
                        'factor_value': yr * 0.1 + mth * 0.01 + dy * 0.001,
                    })

    df = pd.DataFrame(data)
    data_path = PROJECT_ROOT / 'data' / 'test_non_standard.csv'
    df.to_csv(data_path, index=False)

    print(f"✅ Created data with non-standard columns:")
    print(f"   Columns: {list(df.columns)}")
    print(df.head(3))

    # Manual column mapping
    time_config = TimeIndexConfig(
        column_mapping={
            'year': 'fiscal_year',
            'month': 'report_month',
            'day': 'business_day'
        },
        verbose=True
    )

    artefacts = run_pipeline(
        data_path=data_path,
        output_dir='outputs/test_non_standard',
        time_config=time_config,
        entity_col='stock_code',
    )

    print("\n✅ 测试完成！")
    return artefacts


def test_case_4_disable_flattening():
    """Test Case 4: Disable time flattening (use data as-is)."""

    print("\n" + "=" * 70)
    print("测试案例 4: 禁用时间展平")
    print("=" * 70)

    # Create data with standard date column
    data = []
    dates = pd.date_range('2023-01-15', periods=10, freq='D')
    for date in dates:
        for symbol in ['AAPL', 'GOOGL']:
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'symbol': symbol,
                'value': 100.0,
            })

    df = pd.DataFrame(data)
    data_path = PROJECT_ROOT / 'data' / 'test_standard_date.csv'
    df.to_csv(data_path, index=False)

    print(f"✅ Created data with standard date column:")
    print(df.head(3))

    # Disable time flattening
    time_config = TimeIndexConfig(enabled=False)

    artefacts = run_pipeline(
        data_path=data_path,
        output_dir='outputs/test_no_flatten',
        time_config=time_config,
    )

    print("\n✅ 测试完成！")
    return artefacts


def test_case_5_chinese_columns():
    """Test Case 5: Chinese column names (auto-detection)."""

    print("\n" + "=" * 70)
    print("测试案例 5: 中文列名（自动识别）")
    print("=" * 70)

    # Create data with Chinese column names
    data = []
    for year in [2023, 2024]:
        for month in [1, 2]:
            for day in [15, 25]:
                for symbol in ['平安', '茅台']:
                    data.append({
                        '年': year,
                        '月': month,
                        '日': day,
                        '股票代码': symbol,
                        '因子值': year * 0.1 + month * 0.01,
                    })

    df = pd.DataFrame(data)
    data_path = PROJECT_ROOT / 'data' / 'test_chinese.csv'
    df.to_csv(data_path, index=False, encoding='utf-8-sig')

    print(f"✅ Created data with Chinese columns:")
    print(f"   Columns: {list(df.columns)}")
    print(df.head(3))

    # Auto-detection should work for Chinese
    time_config = TimeIndexConfig(verbose=True)

    artefacts = run_pipeline(
        data_path=data_path,
        output_dir='outputs/test_chinese',
        time_config=time_config,
        entity_col='股票代码',
    )

    print("\n✅ 测试完成！")
    return artefacts


def run_all_tests():
    """Run all test cases."""

    print("""
╔══════════════════════════════════════════════════════════════════╗
║   时间展平功能集成测试                                            ║
║   Testing Time Flattening Integration                           ║
╚══════════════════════════════════════════════════════════════════╝
""")

    tests = [
        ("自动时间展平", test_case_1_auto_flatten),
        ("自定义配置", test_case_2_custom_config),
        ("非标准列名", test_case_3_non_standard_columns),
        ("禁用展平", test_case_4_disable_flattening),
        ("中文列名", test_case_5_chinese_columns),
    ]

    results = {}

    for name, test_func in tests:
        try:
            print(f"\n▶ 运行测试: {name}")
            result = test_func()
            results[name] = {'status': 'PASS', 'artefacts': result}
            print(f"✅ {name} - 通过")
        except Exception as e:
            results[name] = {'status': 'FAIL', 'error': str(e)}
            print(f"❌ {name} - 失败: {e}")
            import traceback
            traceback.print_exc()

        if name != tests[-1][0]:
            input("\n按 Enter 继续下一个测试...")

    # Summary
    print("\n" + "=" * 70)
    print("测试摘要")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    total = len(results)

    for name, result in results.items():
        status_symbol = "✅" if result['status'] == 'PASS' else "❌"
        print(f"{status_symbol} {name}: {result['status']}")

    print(f"\n通过率: {passed}/{total} ({passed*100//total}%)")

    if passed == total:
        print("\n🎉 所有测试通过！")
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test time flattening integration")
    parser.add_argument(
        '--test',
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Run specific test case (1-5), or run all if not specified"
    )

    args = parser.parse_args()

    if args.test:
        test_funcs = [
            test_case_1_auto_flatten,
            test_case_2_custom_config,
            test_case_3_non_standard_columns,
            test_case_4_disable_flattening,
            test_case_5_chinese_columns,
        ]
        test_funcs[args.test - 1]()
    else:
        run_all_tests()
