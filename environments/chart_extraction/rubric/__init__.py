import verifiers as vf

from .parsing import cache_parsed_answer, parser
from .rewards.series_points import series_point_count_ratio_reward
from .rewards.series_name_f1 import series_name_f1_reward


rubric = vf.Rubric(parser=parser)
rubric.parallelize_scoring = False
rubric.class_objects["logger"] = rubric.logger

rubric.add_reward_func(cache_parsed_answer, weight=0.0)
rubric.add_reward_func(parser.get_format_reward_func(), weight=1.0)
rubric.add_reward_func(series_name_f1_reward, weight=1.0)
rubric.add_reward_func(series_point_count_ratio_reward, weight=1.0)
