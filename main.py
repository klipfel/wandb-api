import wandb
import pandas as pd
import matplotlib.pyplot as plt
api = wandb.Api()

# run is specified by <entity>/<project>/<run id>
# run = api.run("aklipfel/A1-walking-policy/3c61vcym")
run = api.run("aklipfel/expressive-locomotion/2l5daiwf")

# save the metrics for the run to a csv file
metrics_dataframe = run.history()
# metrics_dataframe = run.scan_history()
print(metrics_dataframe)
metrics_dataframe.to_csv("metrics.csv")

data = pd.read_csv("metrics.csv")
# Only keeps the data from rows where there is an iterationbn update.
data = data.dropna(subset=["Iteration"])
ax = data.plot(
               "Iteration", "average_ll_performance", kind='line', style='b-',
               title="Average reward per step.", legend=None,
               # color='b', linestyle='-'
               )
plt.xlabel("Iteration")
plt.ylabel("Reward")
plt.savefig("average_ll_performance", format='svg')


# ax = data.plot(
#     "Iteration", "average_ll_performance_rup", kind='line', style='b-',
#     title="Average reward per step.", legend=None,
#     # color='b', linestyle='-'
# )
# plt.xlabel("Iteration")
# plt.ylabel("Reward")

iterations = data['Iteration']
print(iterations)


plt.show()
