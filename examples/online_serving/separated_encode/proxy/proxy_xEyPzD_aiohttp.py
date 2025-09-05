# api_proxy.py
# NOT FINISHED YET (HERO, 20250905)
import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional, Dict, Any
from fastapi import FastAPI, Request, HTTPException
import aiohttp
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn
import argparse
import logging
import random

import os
import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

encode_session: Optional[aiohttp.ClientSession] = None
prefill_session: Optional[aiohttp.ClientSession] = None
decode_session: Optional[aiohttp.ClientSession] = None

@app.on_event("startup")
async def startup_event():
    global encode_session, prefill_session, decode_session
    encode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), 
        timeout=aiohttp.ClientTimeout(total=100000))
    prefill_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), 
        timeout=aiohttp.ClientTimeout(total=100000))
    decode_session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), 
        timeout=aiohttp.ClientTimeout(total=100000))

@app.on_event("shutdown")
async def shutdown_event():
    global encode_session, prefill_session, decode_session
    if encode_session:
        await encode_session.close()
    if prefill_session:
        await prefill_session.close()
    if decode_session:
        await decode_session.close()


def has_mm_input(request_data: dict):
    if "messages" not in request_data:
        return False
    for message in request_data["messages"]:  
        if not isinstance(message.get("content"), list):  
            continue
        for content_item in message["content"]:  
            if content_item.get("type") in ["image_url", "audio_url", "input_audio"]:  
                return True 
    return False


async def send_request_to_prefill(prefill_session: aiohttp.ClientSession, p_server_url: str,
                                  req_data: dict, request_id: str):
    """
    Send a request to perfill using a client from the pool.
    # TODO: this is just a temp demo implementation; will need further optimize
    """
    req_data = req_data.copy()
    req_data['kv_transfer_params'] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "remote_engine_id": None,
        "remote_block_ids": None,
        "remote_host": None,
        "remote_port": None
    }
    req_data["stream"] = False
    req_data["max_tokens"] = 1
    if "max_completion_tokens" in req_data:
        req_data["max_completion_tokens"] = 1
    if "stream_options" in req_data:
        del req_data["stream_options"]
    headers = {"x-request-id": request_id}

    response = await prefill_session.post(f"{p_server_url}/v1/chat/completions",
                                                json=req_data,
                                                headers=headers)
    response.raise_for_status()

    return response

async def forward_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    p_server_url: str,
    d_server_url: str,
) -> AsyncIterator[str]:
    
    print(f"request_id: {request_id}", flush=True)
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):
        task1 = asyncio.create_task(
            encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=request_data,
                headers=headers
            )
        )
        try:
            response = await task1  # TODO: try without await?
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail={"error": "Request failed", "message": error_text}
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "message": str(e)}
            )

    # Send request to prefill service
    prefill_response = await send_request_to_service(prefill_client_info, api,
                                                request_data, request_id)

    # Extract the needed fields
    prefill_response_json = prefill_response.json()
    kv_transfer_params = prefill_response_json.get('kv_transfer_params', {})
    if kv_transfer_params:
        request_data["kv_transfer_params"] = kv_transfer_params

    try:
        async with decode_session.post(
            f"{d_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.content.iter_chunked(128):
                if chunk:
                    yield chunk.decode('utf-8', errors='ignore')
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise

async def forward_non_streaming_request(
    request_data: dict,
    request_id: str,
    e_server_url: str,
    p_server_url: str,
    d_server_url: str,
) -> dict:
    # TODO: add EPD adaptability to non streaming request; currently not runnable.
    headers = {"x-request-id": request_id}
    # Skip request to encoder instance if we don't have mm input
    if has_mm_input(request_data):
        # Start request to encode server
        task1 = asyncio.create_task(
            encode_session.post(
                f"{e_server_url}/v1/chat/completions",
                json=request_data,
                headers=headers
            )
        )

        try:
            response = await task1
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(
                    status_code=response.status,
                    detail={"error": "Request failed", "message": error_text}
                )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail={"error": "Internal server error", "message": str(e)}
            )

    try:
        # Make request to decode server
        async with decode_session.post(
            f"{pd_server_url}/v1/chat/completions",
            json=request_data,
            headers=headers
        ) as response2:
            response2.raise_for_status()
            result = await response2.json()
        return result
    except Exception as e:
        logger.error(f"Error in non-streaming: {e}")
        raise

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """Handle chat completion requests."""
    try:
        e_instance = random.randint(0, len(app.state.e_urls) - 1)
        p_instance = random.randint(0, len(app.state.p_urls) - 1)
        d_instance = random.randint(0, len(app.state.p_urls) - 1)
        e_rank = app.state.e_ranks[e_instance]
        p_rank = app.state.pd_ranks[p_instance]
        d_rank = app.state.pd_ranks[p_instance]
        e_server_url = app.state.e_urls[e_instance]
        p_server_url = app.state.p_urls[p_instance]
        d_server_url = app.state.d_urls[p_instance]


        logger.info(f"Matched: E-{e_rank}, P-{p_rank}, D-{d_rank}")

        request_data = await request.json()
        request_id = request.headers.get("x-request-id")
        if not request_id:
            request_id = str(uuid.uuid4())
        request_id = f"{request_id}|{e_rank}|{pd_rank}"
        is_streaming = request_data.get("stream", False)
        if is_streaming:
            return StreamingResponse(
                forward_streaming_request(
                    request_data, request_id, e_server_url, p_server_url, d_server_url),
                media_type="text/event-stream"
            )
        else:
            result = await forward_non_streaming_request(
                request_data, request_id, e_server_url, p_server_url, d_server_url)
            return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    try:
        async with decode_session.get(f"{app.state.d_urls[0]}/v1/models") as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Error fetching models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        async def check_encode():
            try:
                for e_url in app.state.e_urls:
                    async with encode_session.get(f"{e_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
        
        async def check_prefill():
            try:
                for p_url in app.state.p_urls:
                    async with encode_session.get(f"{p_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False

        async def check_decode():
            try:
                for d_url in app.state.d_urls:
                    async with encode_session.get(f"{d_url}/health") as response:
                        response.raise_for_status()
                return True
            except Exception:
                return False
        
        encode_healthy, prefill_healthy, decode_healthy = await asyncio.gather(
            check_encode(), check_prefill(), check_decode(), return_exceptions=True
        )
        
        health_status = {
            "proxy": "healthy",
            "encode_servers": "healthy" if encode_healthy is True else "unhealthy",
            "prefill_servers": "healthy" if prefill_healthy is True else "unhealthy",
            "decode_servers": "healthy" if decode_healthy is True else "unhealthy"
        }
        
        if not (encode_healthy is True and prefill_healthy is True and decode_healthy is True):
            return JSONResponse(content=health_status, status_code=503)
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            content={"proxy": "unhealthy", "error": str(e)},
            status_code=503
        )


### profiling
async def send_profile_cmd(request: Request, req_data, profiler_cmd):
    print(f"profiler_cmd: {profiler_cmd}", flush=True)
    assert profiler_cmd in ["start", "stop"]
    # headers = {"x-request-id": request.req_id}
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    # Send to all prefiller and decoder, leaving iterator in same state.
    tasks = []
    print("About to add tasks...", flush=True)

    # for _ in range(len(app.state.prefill_clients)):
    #     for client in ['prefill', 'decode']:
    #         client_info = get_next_client(request.app, client)

    #         tasks.append(client_info['client'].post(f"/{profiler_cmd}_profile",
    #                                                 json=req_data,
    #                                                 headers=headers))

    
    # for session in [encode_session, decode_session]:
    #     print(f"session: {session}", flush=True)
   #     tasks.append(session.post(f"/{profiler_cmd}_profile",
    #                                 json=req_data,
    #                                 headers=headers))
    #     print(f"tasks: {tasks}", flush=True)

    # TODO: loop through instances instead of hard code in profiling.
    e_instance = random.randint(0, len(app.state.e_urls) - 1)
    pd_instance = 0
    pd_instance2 = 1
    e_server_url = app.state.e_urls[e_instance]
    pd_server_url = app.state.pd_urls[pd_instance]
    pd_server_url2 = app.state.pd_urls[pd_instance2]

    tasks.append(encode_session.post(f"{e_server_url}/{profiler_cmd}_profile",
                                json=req_data,
                                headers=headers))
    tasks.append(decode_session.post(f"{pd_server_url}/{profiler_cmd}_profile",
                                json=req_data,
                                headers=headers))
    tasks.append(decode_session.post(f"{pd_server_url2}/{profiler_cmd}_profile",
                                json=req_data,
                                headers=headers))
    print(f"tasks: {tasks}", flush=True)

    print("About to gather all tasks...", flush=True)
    try:
        responses = await asyncio.gather(*tasks, return_exceptions=True) # CRITICAL CHANGE
        print(f"responses: {responses}", flush=True)
        print("Gather completed.", flush=True)
    except Exception as e:
        print(f"asyncio.gather itself failed: {e}")
        raise

    # Check for exceptions in responses first
    for i, r in enumerate(responses):
        if isinstance(r, Exception):
            print(f"Response {i} was an exception: {r}", flush=True)
            raise r
        else:
            r.raise_for_status()  # Raise HTTP errors for non-exception responses

    a = await responses[0].json(content_type=None)
    print(f"json response: {a}")
    return a


@app.post("/start_profile")
async def start_profile(request: Request):
    try:
        print("start profile in proxy")
        req_data = await request.json()
        # print(f"req_data: {req_data}")
        # raise ValueError(f"req_data: {req_data}")
        # return await send_profile_cmd(request, req_data, "start")
        
        print(f"req_data: {req_data}", flush=True)
        print("?????????????dfsf", flush=True)
        a = await send_profile_cmd(request, req_data, "start")
        print(f"hero: start await send_profile_cmd: {a}")
        return a

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in epd proxy server"
              " - start_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))


@app.post("/stop_profile")
async def stop_profile(request: Request):
    try:
        req_data = await request.json()
        return await send_profile_cmd(request, req_data, "stop")

    except Exception as e:
        import sys
        import traceback
        exc_info = sys.exc_info()
        print("Error occurred in epd proxy server"
              " - stop_profile endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API Proxy for distributed vLLM servers")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Proxy host")
    parser.add_argument("--port", type=int, default=8000, help="Proxy port")

    parser.add_argument("--encode-servers-urls", type=str, required=True,
                       help="URLs of the encode server in comma separated format"
                            "(e.g., \"http://localhost:8001,http://localhost:8002\")")
    
    parser.add_argument("--encode-servers-ranks", type=str, required=True,
                       help="Respective EPD ranks for encode servers in comma-separated format"
                            "(e.g., \"0,1\")")
    
    parser.add_argument("--prefill-servers-urls", type=str, required=True,
                       help="URLs of the prefill servers in comma separated format"
                            "(e.g., \"http://localhost:8003,http://localhost:8004\")")

    parser.add_argument("--prefill-servers-ranks", type=str, required=True,
                       help="Respective EPD ranks for encode servers in comma-separated format"
                            "(e.g., \"2,3\")")

    parser.add_argument("--decode-servers-urls", type=str, required=True,
                       help="URLs of the decode servers in comma separated format"
                            "(e.g., \"http://localhost:8005,http://localhost:8006\")")
    
    parser.add_argument("--decode-servers-ranks", type=str, required=True,
                       help="Respective EPD ranks for encode servers in comma-separated format"
                            "(e.g., \"4,5\")")
    
    args = parser.parse_args()
    app.state.e_urls = args.encode_servers_urls.split(",")
    app.state.p_urls = args.prefill_servers_urls.split(",")
    app.state.d_urls = args.decode_servers_urls.split(",")
    app.state.e_ranks = args.encode_servers_ranks.split(",")
    app.state.p_ranks = args.prefill_servers_ranks.split(",")
    app.state.d_ranks = args.decode_servers_ranks.split(",")
    
    logger.info(f"Starting API proxy on {args.host}:{args.port} with 1 worker")
    logger.info(f"Encode servers: {app.state.e_urls} (respective ranks {app.state.e_ranks})")
    logger.info(f"Prefill server: {app.state.p_urls} (respective ranks {app.state.p_ranks})")
    logger.info(f"Decode server: {app.state.d_urls} (respective ranks {app.state.d_ranks})")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=False,
        loop="uvloop"
    )