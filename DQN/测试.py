class CliffEnv():
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.env = self.init()

    def move(self, place, act):
        x_next = place[0] + act[0]
        y_next = place[1] + act[1]
        return [x_next, y_next]

    def init(self):
        action = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        P = []
        for i in range(self.row):
            for j in range(self.col):
                personal_state = []
                for k in range(len(action)):
                    s = i * self.col + j
                    state = [i, j]
                    state_next = self.move(state, action[k])
                    done = False
                    if i == self.row - 1 and 0 < j < self.col:
                        personal_state.append([s, action[k], 0, s, True])
                        continue
                    x_next = min(self.row - 1, max(0, state_next[0]))
                    y_next = min(self.col - 1, max(0, state_next[1]))
                    s_next = x_next * self.col + y_next
                    if x_next == self.row - 1 and 0 < y_next < self.col - 1:
                        personal_state.append([state, action[k], -100, s_next, False])
                        continue
                    if x_next == self.row - 1 and y_next == self.col - 1:
                        personal_state.append([state, action[k], 0, s_next, False])
                        continue
                    else:
                        personal_state.append([state, action[k],-1, s_next, False])
                P.append(personal_state)
        return P
if '__main__' == __name__:
    cliff = CliffEnv(4, 12)
    cliff.init()
    print(cliff.env)