"""
Lab #3: Baseline Chatbot vs ReAct Agent
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP, get_flight_info, get_weather_forecast

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

AIRPORT_CODES = ("HAN", "SGN", "DAD")

class ChatbotBaseline:
    """Baseline LLM Chatbot (Không sử dụng ReAct Loop hay Tools)"""

    def query(self, user_input: str) -> Dict[str, Any]:
        answer = (
            "Tôi không có khả năng tra cứu dữ liệu chuyến bay hay thời tiết theo thời gian thực. "
            "Dựa trên kiến thức chung, bạn nên tự kiểm tra trên website hãng bay và ứng dụng thời tiết "
            "để có thông tin chính xác nhất trước khi đặt vé."
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": []
        }


class ReActAgent:
    """ReAct Agent có sử dụng Thought-Action-Observation Loop"""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )

    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        iteration = 0
        flight_result: Optional[List[Dict[str, Any]]] = None
        weather_result: Optional[Dict[str, Any]] = None

        needs_flight = self._needs_flight(user_input)
        needs_weather = self._needs_weather(user_input)

        while iteration < self.max_iterations:
            iteration += 1

            if needs_flight and flight_result is None:
                llm_output = self._build_flight_action(user_input)
            elif needs_weather and weather_result is None:
                llm_output = self._build_weather_action(user_input, flight_result)
            else:
                answer = self._compose_final_answer(needs_flight, needs_weather, flight_result, weather_result)
                llm_output = f"Thought: Đã thu thập đủ thông tin cần thiết để trả lời khách hàng.\nFinal Answer: {answer}"

            thought = self._extract_thought(llm_output)

            if "Final Answer:" in llm_output:
                answer = llm_output.split("Final Answer:", 1)[1].strip()
                self.trace.append({
                    "iteration": iteration,
                    "thought": thought,
                    "action": None,
                    "observation": None,
                    "final_answer": answer
                })
                return {
                    "status": "completed",
                    "answer": answer,
                    "iterations": iteration,
                    "trace": self.trace
                }

            action, observation = self._execute_action(llm_output)

            if action is not None:
                tool_name = str(action.get("name", "")).strip().lower()
                if tool_name == "get_flight_info":
                    flight_result = observation
                elif tool_name == "get_weather_forecast":
                    weather_result = observation

            self.trace.append({
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation
            })

            # Nếu chỉ có một loại thông tin cần thu thập, tổng hợp câu trả lời ngay
            # trong cùng iteration. Nếu cần cả hai (chuyến bay + thời tiết), dành
            # riêng một iteration cuối để tổng hợp Final Answer.
            if not (needs_flight and needs_weather):
                if flight_result is not None or weather_result is not None:
                    answer = self._compose_final_answer(needs_flight, needs_weather, flight_result, weather_result)
                    self.trace[-1]["final_answer"] = answer
                    return {
                        "status": "completed",
                        "answer": answer,
                        "iterations": iteration,
                        "trace": self.trace
                    }

        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành yêu cầu trong số bước tối đa cho phép.",
            "iterations": iteration,
            "trace": self.trace
        }

    # ------------------------------------------------------------------
    # "Thought" generation (mô phỏng LLM suy luận dựa trên câu hỏi khách hàng)
    # ------------------------------------------------------------------
    def _needs_flight(self, text: str) -> bool:
        codes = set(re.findall(r"\b(HAN|SGN|DAD)\b", text))
        return len(codes) >= 2

    def _needs_weather(self, text: str) -> bool:
        keywords = ["thời tiết", "mặc gì", "trang phục"]
        text_lower = text.lower()
        return any(k in text_lower for k in keywords)

    def _build_flight_action(self, user_input: str) -> str:
        origin, destination = self._parse_route(user_input)
        max_price = self._parse_max_price(user_input)
        action = {
            "name": "get_flight_info",
            "args": {"origin": origin, "destination": destination, "max_price": max_price}
        }
        thought = (
            f"Khách hàng cần tìm chuyến bay từ {origin} đến {destination} "
            f"với ngân sách tối đa {max_price:,}đ. Tôi sẽ gọi tool get_flight_info."
        )
        return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}"

    def _build_weather_action(self, user_input: str, flight_result: Optional[List[Dict[str, Any]]]) -> str:
        destination = flight_result[0]["destination"] if flight_result else None
        city_code = self._parse_weather_city(user_input) or destination
        action = {"name": "get_weather_forecast", "args": {"city_code": city_code}}
        thought = (
            f"Khách hàng cần thông tin thời tiết tại {city_code} để chuẩn bị trang phục. "
            "Tôi sẽ gọi tool get_weather_forecast."
        )
        return f"Thought: {thought}\nAction: {json.dumps(action, ensure_ascii=False)}"

    def _extract_thought(self, llm_output: str) -> str:
        first_line = llm_output.split("\n", 1)[0]
        return first_line.replace("Thought:", "").strip()

    def _execute_action(self, llm_output: str) -> Tuple[Optional[Dict[str, Any]], Any]:
        match = re.search(r"Action:\s*(\{.*\})", llm_output, re.DOTALL)
        if not match:
            return None, "Observation: Invalid JSON format"

        try:
            action = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            return None, "Observation: Invalid JSON format"

        tool_name = str(action.get("name", "")).strip().lower()
        tool_fn = TOOL_MAP.get(tool_name)
        if tool_fn is None:
            return action, f"Error: Không tìm thấy tool '{tool_name}'"

        args = action.get("args", {})
        observation = tool_fn(**args)
        return action, observation

    # ------------------------------------------------------------------
    # Parsing tiện ích (trích xuất tham số từ câu hỏi tiếng Việt)
    # ------------------------------------------------------------------
    def _parse_route(self, text: str) -> Tuple[Optional[str], Optional[str]]:
        codes = re.findall(r"\b(HAN|SGN|DAD)\b", text)
        origin = codes[0] if len(codes) >= 1 else None
        destination = codes[1] if len(codes) >= 2 else None
        return origin, destination

    def _parse_max_price(self, text: str) -> int:
        text_lower = text.lower()
        million_match = re.search(r"(\d+(?:[.,]\d+)?)\s*triệu", text_lower)
        if million_match:
            value = float(million_match.group(1).replace(",", "."))
            return int(value * 1_000_000)

        k_match = re.search(r"(\d+)\s*k\b", text_lower)
        if k_match:
            return int(k_match.group(1)) * 1000

        return 5_000_000

    def _parse_weather_city(self, text: str) -> Optional[str]:
        idx = text.lower().find("thời tiết")
        codes = list(re.finditer(r"\b(HAN|SGN|DAD)\b", text))
        if idx != -1:
            after = [c.group(1) for c in codes if c.start() >= idx]
            if after:
                return after[0]
        if codes:
            return codes[-1].group(1)
        return None

    # ------------------------------------------------------------------
    # Tổng hợp Final Answer
    # ------------------------------------------------------------------
    def _compose_final_answer(
        self,
        needs_flight: bool,
        needs_weather: bool,
        flight_result: Optional[List[Dict[str, Any]]],
        weather_result: Optional[Dict[str, Any]]
    ) -> str:
        parts = []

        if needs_flight:
            if flight_result:
                flight_lines = [
                    f"- {f['flight_number']} ({f['airline']}) khởi hành {f['departure_time']}, giá {f['price_vnd']:,}đ"
                    for f in flight_result
                ]
                parts.append("Các chuyến bay phù hợp:\n" + "\n".join(flight_lines))
            else:
                parts.append("Hiện không tìm thấy chuyến bay nào phù hợp với yêu cầu của bạn.")

        if needs_weather:
            if weather_result and "error" not in weather_result:
                parts.append(
                    f"Thời tiết tại {weather_result['city']}: {weather_result['temperature_c']}°C, "
                    f"{weather_result['condition']}. Gợi ý trang phục: {weather_result['recommendation']}"
                )
            else:
                parts.append("Không tìm thấy thông tin thời tiết cho địa điểm này.")

        if not needs_flight and not needs_weather:
            parts.append(
                "Đây là câu hỏi ngoài phạm vi tra cứu dữ liệu chuyến bay/thời tiết. "
                "Vui lòng liên hệ tổng đài chăm sóc khách hàng Vinpearl để được tư vấn chi tiết về chính sách đổi trả vé."
            )

        return " ".join(parts)


def main():
    user_query = "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?"

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(json.dumps(chatbot.query(user_query), indent=2, ensure_ascii=False))

    print("\n=== RUNNING REACT AGENT ===")
    agent = ReActAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:", json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
