# Evaluation: 30 episodes per policy and city group, difficulty 1.0

## Outcome (success with 95 % Wilson interval)

| policy | cities | n  | success              | collision | stall | lost | timeout |
|--------|--------|----|----------------------|-----------|-------|------|---------|
| expert | seen   | 30 | 60.0% [42.3%..75.4%] | 30.0%     | 6.7%  | 0.0% | 3.3%    |
| model  | seen   | 30 | 56.7% [39.2%..72.6%] | 36.7%     | 0.0%  | 0.0% | 6.7%    |
| expert | unseen | 30 | 56.7% [39.2%..72.6%] | 30.0%     | 13.3% | 0.0% | 0.0%    |
| model  | unseen | 30 | 53.3% [36.1%..69.8%] | 46.7%     | 0.0%  | 0.0% | 0.0%    |

## What was hit (share of all episodes)

| policy | cities | wall  | parked | car   | bus  | pedestrian |
|--------|--------|-------|--------|-------|------|------------|
| expert | seen   | 16.7% | 0.0%   | 3.3%  | 3.3% | 6.7%       |
| model  | seen   | 16.7% | 0.0%   | 13.3% | 0.0% | 6.7%       |
| expert | unseen | 26.7% | 0.0%   | 0.0%  | 0.0% | 3.3%       |
| model  | unseen | 10.0% | 0.0%   | 23.3% | 3.3% | 10.0%      |

## Rule violations (events per episode; viol_any = episodes with >= 1; speeding / wrong side = time share)

| policy | cities | red_light | yellow_light | stop_sign | yield | zebra | viol_any | speeding | wrong side |
|--------|--------|-----------|--------------|-----------|-------|-------|----------|----------|------------|
| expert | seen   | 0.000     | 0.000        | 0.000     | 0.000 | 0.000 | 0.0%     | 0.0%     | 2.2%       |
| model  | seen   | 0.333     | 0.000        | 0.067     | 0.067 | 0.133 | 36.7%    | 0.1%     | 4.5%       |
| expert | unseen | 0.033     | 0.000        | 0.000     | 0.000 | 0.000 | 3.3%     | 0.0%     | 0.4%       |
| model  | unseen | 0.200     | 0.000        | 0.000     | 0.000 | 0.067 | 23.3%    | 0.0%     | 0.7%       |

## Driving manners

| policy | cities | AEB time | needless slow | speed km/h | steer jerk | lane dev m | route done |
|--------|--------|----------|---------------|------------|------------|------------|------------|
| expert | seen   | 0.5%     | 15.4%         | 16.4       | 0.0116     | 0.36       | 78.4%      |
| model  | seen   | 0.7%     | 16.8%         | 17.9       | 0.0232     | 0.63       | 70.7%      |
| expert | unseen | 0.9%     | 9.7%          | 16.0       | 0.0075     | 0.32       | 76.3%      |
| model  | unseen | 1.1%     | 12.9%         | 20.9       | 0.0220     | 0.55       | 71.8%      |

## Memorising check (seen -> unseen cities)

| policy | success seen | success unseen | change  |
|--------|--------------|----------------|---------|
| expert | 60.0%        | 56.7%          | -3.3 pp |
| model  | 56.7%        | 53.3%          | -3.3 pp |

A drop of more than ~10 pp (beyond the intervals above) would mean the map was memorised.
