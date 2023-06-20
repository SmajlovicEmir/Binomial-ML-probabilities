import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats.mstats
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from BinaryTree import BinomialTree, Node
from scipy.stats import reciprocal, lognorm
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics._scorer import make_scorer
from scipy import stats
from scipy.special import binom
import itertools

# https://users.physics.ox.ac.uk/~Foot/Phynance/Binomial2013.pdf
# https://web.ma.utexas.edu/users/mcudina/bin_tree_lognormal.pdf


class BinomialPriceTree(BinomialTree):
    def __init__(self, price, up_jump, down_jump, prob_u, prob_d, func_price, depth=10):
        super().__init__(depth)
        self.price = price
        self.up_jump = up_jump
        self.down_jump = down_jump
        self.calc_price_function = func_price
        self.probability_up = prob_u
        self.probability_down = prob_d

    def calculate_prices(self):
        for a, branch in enumerate(self.tree):
            for b, node in enumerate(branch):
                node.set_data(self.calc_price_function(self.price, self.up_jump, self.down_jump, a, b))

    def calculate_expected_price(self, at_depth=None):
        if at_depth is None:
            at_depth = len(self.tree) - 1
        temp_list = []
        for y, node in enumerate(self.tree[at_depth]):
            temp_list.append(binom(at_depth, y) * (self.probability_up ** (at_depth - y) * self.probability_down ** y) * node.get_data()[0][0])
        return np.sum(temp_list)

    def print_tree(self):
        mesh = np.full((len(self.tree), 2 * len(self.tree) - 1), 0)
        for xy, z in zip(self.sorted_coo, self.flat_tree):
            print(z.get_data()[0][0], " ", end="")
            mesh[xy[0]][xy[1]] = z.get_data()[0][0]
        mesh = mesh.T
        print(mesh)
        plt.plot(mesh)


def calculate_future_price(price, proba_up, proba_down, iter_1, iter_2):
    # formula for value at i-th step s_i_j = S_0 * u^i * d^(i-j)
    print(f"up: {iter_1 - iter_2}, down: {iter_2}")
    return price * (proba_up ** (iter_1 - iter_2)) * (proba_down ** iter_2)


def calculate_parameters_binomial(price_window, step_size=1, r=0):
    up_jump = np.exp((price_window.std().iloc[0]) * np.sqrt(step_size / 10))
    down_jump = 1 / up_jump
    probability_up = (np.e ** (r * step_size) - down_jump) / (up_jump - down_jump)
    probability_down = 1 - probability_up
    return up_jump, down_jump, probability_up, probability_down


def create_feature_label_set(price_df):
    temp_list = []
    for s in range(price_df.shape[0] - 10):
        temp_window = price_df.iloc[s:s + 10].reset_index()
        temp_window = temp_window.T
        temp_window.drop("index", axis=0, inplace=True)
        temp_window[10] = temp_window[8]
        temp_window[11] = temp_window[9]
        temp_list.append(temp_window)
    return temp_list


def binomial_prediction_error(y_true, y_pred):
    # the price at s_t we are trying to estimate is equal. s_t = pSu + (1-q)Sd, u and d we get from the above function
    s_t = (y_pred[:, 0] * y_true[0][0] * u + y_true[0][0] * y_pred[:, 1] * d)
    return tf.reduce_mean(tf.square(y_true[0][1] - s_t))


def neg_binomial_prediction_error(y_true, y_pred):
    # the price at s_t we are trying to estimate is equal. s_t = pSu + (1-q)Sd, u and d we get from the above function
    s_t = (y_pred[:, 0] * y_true[0][0] * u + y_true[0][0] * y_pred[:, 1] * d)
    print(-1 * tf.reduce_mean(tf.square(y_true[0][1] - s_t)))
    return -1 * (tf.reduce_mean(tf.square(y_true[0][1] - s_t))).numpy()


def build_model(hidden_layers=1, no_of_neurons=100, lr=25e-3):
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten())
    model.add(keras.layers.InputLayer(input_shape=(None, x_windows.shape[1])))
    for layer in range(hidden_layers):
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(no_of_neurons, activation="relu", kernel_initializer="he_normal"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(loss=binomial_prediction_error, optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


np.random.seed(42)
random_normal = np.exp(np.random.randn(250))  # 250 samples of the normal distribution with mean = 0 and std: 1
mean = np.full((1, 250), random_normal.mean())
std_dev = np.full((1, 250), random_normal.std())
x = np.linspace(np.min(random_normal), np.max(random_normal), 250)
plt.plot(x, random_normal, label="exponential random normal variable")
plt.plot(x, mean.T, label="mean")
plt.plot(x, std_dev.T, label="standard deviation")
plt.fill_between(x, 0, random_normal.std(), alpha=0.2)
plt.fill_between(x, random_normal.std(), 2 * random_normal.std(), alpha=0.2)
plt.fill_between(x, 2 * random_normal.std(), 3 * random_normal.std(), alpha=0.2)
plt.grid()
plt.legend()
plt.show()

nvidia_data = pd.read_csv(r"C:\Users\emirs\Desktop\PyProjects\RandomForrestBinomialOptionPricing\NVDA.csv")
nvidia_open_prices = nvidia_data.loc[:, "Open"]

rolling_average_openprices = nvidia_open_prices.rolling(10).mean()
rolling_sigma_openprices_upper = nvidia_open_prices.rolling(10).std() + rolling_average_openprices
rolling_sigma_openprices_lower = rolling_average_openprices - nvidia_open_prices.rolling(10).std()
nvidia_open_prices[9:].plot(label="Actual Prices")
rolling_sigma_openprices_upper.plot(label="Mean + Std.Dev")
rolling_sigma_openprices_lower.plot(label="Mean - Std.Dev")
rolling_average_openprices.plot(label="Mean")
plt.legend()
plt.grid()
plt.show()
stdrdized_prices = (nvidia_open_prices - nvidia_open_prices.mean()) / nvidia_open_prices.std()

train_set_open_prices = nvidia_open_prices[:101]
train_open_prices_return = np.log(train_set_open_prices.iloc[1:]) / np.log(train_set_open_prices.iloc[:-1])
prices_t_1 = train_set_open_prices.iloc[:-1].reset_index()
prices_t_0 = train_set_open_prices.iloc[1:].reset_index()
prices_t_0.drop("index", axis="columns", inplace=True)
prices_t_1.drop("index", axis="columns", inplace=True)

p = log_returns = prices_t_1 / prices_t_0
std_bins = []
for i in range(-3, 4):
    if i == 0:
        continue
    std_bins.append(i * p.std().iloc[0])
print(std_bins)

print("interval sum", np.sum(p < std_bins[1]))
plt.hist(p, density=True, bins=6, ec="black")
plt.show()
print(scipy.stats.mstats.normaltest(log_returns))
for_q_plot = np.array(p.iloc[:])
fig = plt.figure()
print("for_q_plot: ")
print(for_q_plot)
ax = fig.add_subplot(111)
stats.probplot(for_q_plot.ravel(), sparams=(p.std().iloc[0]), dist=lognorm, plot=ax)
ax.set_title(f"Q-Q Plot for a log-normal pdf, with the standard deviation of: {p.std().iloc[0]:.2f}")
plt.show()

fig, ax = plt.subplots()
ax.semilogy(log_returns, base=np.e)
plt.show()

u, d, p_u, p_d = calculate_parameters_binomial(log_returns, 10)

labeled_set = np.array(create_feature_label_set(nvidia_open_prices))
labeled_set = labeled_set.reshape(-1, 12)
pd_labeled_set = pd.DataFrame(labeled_set)

x_train_full, x_test = train_test_split(labeled_set, train_size=0.8, test_size=0.2)
x_train, x_val = train_test_split(x_train_full, train_size=0.6, test_size=0.2)
x_windows, y_prices = x_train[:, :-2], x_train[:, -2:]
x_val_windows, y_val_prices = x_val[:, :-2], x_val[:, -2:]

"""
class BinomialTreeLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        s_t = (y_pred[:, 0] * y_true[0][0] * u + y_true[0][0] * y_pred[:, 1] * d)
        return s_t
"""

x_windows = x_windows.reshape(78, 10)
kerasreg_cv = keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {
    "hidden_layers": np.arange(1, 10),
    "no_of_neurons": np.arange(1, 100),
    "lr": reciprocal(3e-4, 3e-2),
}

neg_bin_scorrer = make_scorer(neg_binomial_prediction_error, greater_is_better=True)

rnd_search_cv = RandomizedSearchCV(kerasreg_cv, param_distribs, n_iter=10, cv=3, scoring=neg_bin_scorrer)
rnd_search_cv.fit(x_windows, y_prices, epochs=1, validation_data=(x_val_windows, y_val_prices),
                  verbose=2, callbacks=[keras.callbacks.EarlyStopping(patience=10)])
print(f"Best parameters: {rnd_search_cv.best_params_},\n best_score: {rnd_search_cv.best_score_}")

bin_model = rnd_search_cv.best_estimator_.model
print(bin_model.summary())
print("model probabilities:")
print(bin_model.predict(x_test[:, :-2]))

price_probas_bin_ml = bin_model.predict(x_test[:, :-2])
predicted_prices_ml = price_probas_bin_ml * x_test[:, -2:-1] * [u, d]
predicted_prices_ml = [np.sum(x) for x in predicted_prices_ml]

predicted_prices_classic_fixed_probabilities = x_test[:, -2:-1] * [u, d] * [p_u, p_d]
predicted_prices_classic_fixed_probabilities = [np.sum(x) for x in predicted_prices_classic_fixed_probabilities]

ax1 = plt.subplot(411)
ax1.plot(predicted_prices_classic_fixed_probabilities, label="classic approach")
ax1.plot(predicted_prices_ml, label="set-estimated probabilities")
ax1.plot(x_test[:, -1:], label="true prices")
ax1.set_title("All predictions vs. true")
plt.grid()
plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")

MSE_fixed_estimated = np.mean(np.square(np.array(predicted_prices_classic_fixed_probabilities) - predicted_prices_ml))

ax2 = plt.subplot(412)
ax2.plot(predicted_prices_classic_fixed_probabilities, label="classic approach")
ax2.plot(predicted_prices_ml, label="set-estimated probabilities")
ax2.set_title(f"Estimated vs. Fixed probabilities, MSE: {MSE_fixed_estimated:.2f}")
plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
plt.grid()

MSE_true_estimated = np.mean(np.square(x_test[:, -1:] - predicted_prices_ml))

ax3 = plt.subplot(413)
ax3.plot(predicted_prices_ml, label="set-estimated probabilities")
ax3.plot(x_test[:, -1:], label="true prices")
ax3.set_title(f"Estimated vs. True prices, MSE: {MSE_true_estimated:.2f}")
plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
plt.grid()

MSE_true_fixed = np.mean(np.square(x_test[:, -1:] - np.array(predicted_prices_classic_fixed_probabilities)))

ax4 = plt.subplot(414)
ax4.plot(predicted_prices_classic_fixed_probabilities, label="classic approach")
ax4.plot(x_test[:, -1:], label="true prices")
ax4.set_title(f"Fixed vs. True prices, MSE: {MSE_true_fixed:.2f}")
plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
plt.grid()

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.789,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()


all_probs = bin_model.predict(x_test[:1, :-2])
probability_up = all_probs[0][0]
probability_down = all_probs[0][1]
print(f"Prob up: {probability_up}, Prob down: {probability_down}")
price_t0 = x_test[:1, -2:-1]
print(price_t0)
predictedProbabilityPrice = BinomialPriceTree(price_t0, u, d, probability_up, probability_down, calculate_future_price)
predictedProbabilityPrice.calculate_prices()
predictedProbabilityPrice.plot_tree()
predictedProbabilityPrice.print_tree()
print("Expected price: ", predictedProbabilityPrice.calculate_expected_price())
