from typing import Optional

from chatbot.states import State, Initial

if __name__ == "__main__":
    state = Initial()  # type: Optional[State]
    while state:
        state = state.run()
