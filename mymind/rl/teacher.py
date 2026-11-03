"""DeepSeek 真实开发轨迹采集入口。"""
import asyncio
from rl.evaluate import main

if __name__ == "__main__":
    asyncio.run(main(teacher=True))
