from test_pkg import train
print(f"train is: {train}")
try:
    train()
except Exception as e:
    print(e)

