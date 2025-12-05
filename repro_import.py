try:
    from friedman_net import train
    print(f"Successfully imported train: {train}")
    print(f"Type of train: {type(train)}")
except Exception as e:
    print(f"Error importing train: {e}")

try:
    from friedman_net import MarketLayer
    print(f"Successfully imported MarketLayer: {MarketLayer}")
except Exception as e:
    print(f"Error importing MarketLayer: {e}")

