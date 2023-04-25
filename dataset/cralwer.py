import asyncio
import aiohttp

CONCURRENT_DOWNLOADS = 8

base_url = "http://www.tng-project.org/api/Illustris-1/snapshots/135/subhalos/"

async def download_url(url, session, semaphore,headers,id):
    async with semaphore:
        async with session.get(url,headers=headers) as response:
            if response.status == 404:
                print(f"Error 404: {id} not found. Skipping download.")
            else:
                file_name = "./Illustris-1/{}.png".format(id)
                with open(file_name, 'wb') as f:
                    while True:
                        chunk = await response.content.read(1024)
                        if not chunk:
                            break
                        f.write(chunk)
            return await response.release()

async def main(start_id):

    headers = {"api-key": "#"}
    connector = aiohttp.TCPConnector(limit=8)
    async with aiohttp.ClientSession(connector=connector) as session:
        semaphore = asyncio.Semaphore(CONCURRENT_DOWNLOADS)
        tasks = []
        for i in range(start_id,start_id+5000):
            url = "http://www.tng-project.org/api/Illustris-1/snapshots/135/subhalos/{}/stellar_mocks/image_subhalo.png".format(str(i))
            tasks.append(asyncio.ensure_future(download_url(url, session, semaphore,headers,str(i))))

        print("tasks launched")

        await asyncio.gather(*tasks)

if __name__ == '__main__':
    total_subhalos=4366546
    partion = 1000
    for i in range(0,180):
        try:
            asyncio.run(main(i*partion))
        except:
            continue
