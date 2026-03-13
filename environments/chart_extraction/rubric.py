import verifiers as vf


parser = vf.XMLParser(["answer"])
answer_format_reward = parser.get_format_reward_func()


rubric = vf.Rubric(parser=parser)
rubric.add_reward_func(answer_format_reward, weight=1.0)
