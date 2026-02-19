import os
import openai

def test_openai_api(api_key="sk-QzOeTbtA4khBqsJ7fJ0WL3bJREi7pjVjvrRhS9DGjANVZt6w", base_url=None):
    # 如果没有传递 api_key，默认从环境变量读取
    api_key = api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("API key is required.")
        return
    
    # 如果有提供 base_url，设置 base_url
    base_url = base_url or os.getenv("OPENAI_BASE_URL")
    print(base_url)
    if base_url:
        openai.api_base = base_url
    
    try:
        # 设置 API 密钥
        openai.api_key = api_key
        
        # 测试请求：发送一个简单的聊天请求
        response = openai.chat.completions.create(
            model="gpt-5-nano",  # 或者你可以替换成其他支持的模型，如 "gpt-4"
            messages=[
                {"role": "user", "content": "who are you?"}
            ],
            max_completion_tokens=40
        )
        
        # 打印返回的完整响应
        print("Response from OpenAI API:")
        print(response)
        
        # 打印响应中的生成内容
        print("\nGenerated text:")
        print(response['choices'][0]['message']['content'])
        
    except openai.APIConnectionError as e:
        # 捕获 API 连接错误
        print(f"OpenAI API connection error: {e}")
    except openai.OpenAIError as e:
        # 捕获其他 OpenAI 错误
        print(f"OpenAI API error: {e}")

if __name__ == "__main__":
    # 可选：你可以指定base_url，或者通过环境变量 OPENAI_BASE_URL 设置
    test_openai_api(base_url="https://new.apifast.top/")  # 或者使用你自己的 base_url
