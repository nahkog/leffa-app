import redis.asyncio as redis

redis_client = redis.Redis(
    host="redis-13558.crce175.eu-north-1-1.ec2.redns.redis-cloud.com",
    port=13558,
    username="default",
    password="mEAvbHViJCeEsNxrl0MnmdtMC5Rg6pcV",
    decode_responses=True
)

if __name__ == "__main__":
    import asyncio
    
    async def test():
        await redis_client.set("test_key", "hello")
        val = await redis_client.get("test_key")
        print("Redis test:", val)

    asyncio.run(test())
