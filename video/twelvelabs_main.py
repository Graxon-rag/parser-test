from twelvelabs import AsyncTwelveLabs
from twelvelabs.types import VideoContext_AssetId
from dotenv import load_dotenv
from pathlib import Path
import asyncio
import os

load_dotenv()


async def main():
    client = AsyncTwelveLabs(api_key=os.getenv("TWELVELABS_API_KEY"))

    video_path = Path("/home/avvk/Graxon/Graxon/parser/test_data/IMG_3509.MOV")

    print("Uploading video...")
    video_obj = await client.assets.create(
        file=video_path.read_bytes(),
        method="direct",
    )

    if video_obj.id is None:
        raise ValueError("Video upload failed")

    print(f"Video uploaded successfully. Asset ID: {video_obj.id}")

    print("Creating analysis task...")
    task = await client.analyze_async.tasks.create(
        model_name="pegasus1.5",
        video=VideoContext_AssetId(asset_id=video_obj.id),
        prompt="Summarise this video based in segment like on this segment this happened and give me transcript of video",
        analysis_mode="general"
    )

    if task.task_id is None:
        raise ValueError("Task creation failed")

    print(f"Task created successfully. Task ID: {task.task_id}")

    print("Polling task status...")
    # The API returns "ready" on success or "failed" on failure. 
    while task.status not in ["ready", "failed"]:
        await asyncio.sleep(5)
        # Use analyze_async to retrieve the analysis task specifically
        task = await client.analyze_async.tasks.retrieve(task.task_id)
        print(f"Current status: {task.status}")

    if task.status == "failed":
        print("Task failed!")
        # The docs state that if the status is failed, there will be an error object/message
        if hasattr(task, 'error'):
            print(f"Error details: {task.error}")
        return

    print("\n=== Analysis Complete ===")

    # When status is "ready", the payload includes a `result` object containing `data`
    if hasattr(task, 'result') and task.result:
        # For general analysis, the generated text response lives here
        print(task.result.data)
    else:
        print("Task completed, but no result data was returned.")

if __name__ == "__main__":
    asyncio.run(main())
