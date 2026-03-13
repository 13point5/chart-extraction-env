import verifiers as vf

from .parsing import cache_parsed_answer, parser
from .rewards.series_points import series_point_count_ratio
from .rewards.series_point_values import series_point_value
from .rewards.series_name_f1 import series_name_f1


rubric = vf.Rubric(parser=parser)
rubric.parallelize_scoring = False
rubric.class_objects["logger"] = rubric.logger
format = parser.get_format_reward_func()
format.__name__ = "format"

rubric.add_reward_func(cache_parsed_answer, weight=0.0)
rubric.add_reward_func(format, weight=1.0)
rubric.add_reward_func(series_name_f1, weight=1.0)
rubric.add_reward_func(series_point_count_ratio, weight=1.0)
rubric.add_reward_func(series_point_value, weight=1.0)
