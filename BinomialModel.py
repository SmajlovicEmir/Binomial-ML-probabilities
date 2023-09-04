import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats.mstats
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from BinaryTree import BinomialTree
from scipy.stats import reciprocal, lognorm, randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics._scorer import make_scorer
from scipy import stats
from scipy.special import binom
import itertools
from typing import Optional, Any
from copy import deepcopy
import os

print(keras.__version__)
print(tf.__version__)


# https://users.physics.ox.ac.uk/~Foot/Phynance/Binomial2013.pdf
# https://web.ma.utexas.edu/users/mcudina/bin_tree_lognormal.pdf


class BinomialPriceTree(BinomialTree):
    def __init__(self, price: float, up_jump: float, down_jump: float, prob_u: float, prob_d: float,
                 func_price: callable, depth: int = 10):
        super().__init__(depth)
        self.price = price
        self.up_jump = up_jump
        self.down_jump = down_jump
        self.calc_price_function = func_price
        self.probability_up = prob_u
        self.probability_down = prob_d

    def calculate_prices(self) -> None:
        for a, branch in enumerate(self.tree):
            for b, node in enumerate(branch):
                node.data = self.calc_price_function(self.price, self.up_jump, self.down_jump, a, b)

    def calculate_expected_price(self, at_depth: int = None) -> npt.NDArray[float]:
        if at_depth is None:
            at_depth = len(self.tree) - 1
        temp_list = []
        for y, node in enumerate(self.tree[at_depth]):
            temp_list.append(binom(at_depth, y) * (self.probability_up ** (at_depth - y) * self.probability_down ** y) *
                             node.data)
        return np.sum(temp_list)

    def print_tree(self, *, return_matrix: bool = None) -> Optional[list[float]]:
        mesh = np.full((len(self.tree), 2 * len(self.tree) - 1), 0, dtype=float)
        for xy, z in zip(self.sorted_coo, self.flat_tree):
            mesh[xy[0]][xy[1]] = round(z.data, 2)
        mesh = mesh.T
        print(mesh)
        if return_matrix:
            return mesh

    def plot_tree(self) -> None:
        if not self.is_set_up:
            self.set_up()
        plt.plot(self.row_coo, self.y_coo_sorted, "ob")
        plt.yticks(np.arange(self.depth * 2 - 1))
        plt.xticks(np.arange(self.depth))
        temp_tree = deepcopy(self.tree)
        for branch in temp_tree:
            branch.reverse()

        temp_tree = list(itertools.chain(*temp_tree))

        for coo, node in zip(self.sorted_coo, temp_tree):
            plt.annotate(round(node.data, 2), (coo[0] - .05, coo[1] + .25))

        for x_coo in range(len(self.tree) - 1):
            for y in range(x_coo + 1):
                plt.plot([self.tree[x_coo][y].coo[0], self.tree[x_coo + 1][y].coo[0]],
                         [self.tree[x_coo][y].coo[1], self.tree[x_coo + 1][y].coo[1]], "k-")
                plt.plot([self.tree[x_coo][y].coo[0], self.tree[x_coo + 1][y + 1].coo[0]],
                         [self.tree[x_coo][y].coo[1], self.tree[x_coo + 1][y + 1].coo[1]], "k-")

        plt.show()

    def get_plot(self, x_offset: int) -> Any:
        if not self.is_set_up:
            print("setting up...")
            self.set_up()
        temp_tree = deepcopy(self.tree)
        temp_coo = self.sorted_coo[:]
        for branch in temp_tree:
            branch.reverse()
        temp_fig, temp_ax = plt.subplots()

        temp_tree = list(itertools.chain(*temp_tree))
        offset_coo = map(lambda coo_lambda: (coo_lambda[0] + x_offset, coo_lambda[1]), temp_coo)
        for coo, node in zip(offset_coo, temp_tree):
            temp_ax.plot(coo[0], node.data, "ob")
            temp_ax.annotate(round(node.data, 2), (coo[0] - .10, node.data + 1))

        for x_coo in range(len(self.tree) - 1):
            for y in range(x_coo + 1):
                temp_ax.plot([self.tree[x_coo][y].coo[0] + x_offset, self.tree[x_coo + 1][y].coo[0] + x_offset],
                             [self.tree[x_coo][y].data, self.tree[x_coo + 1][y].data], "k-")
                temp_ax.plot([self.tree[x_coo][y].coo[0] + x_offset, self.tree[x_coo + 1][y + 1].coo[0] + x_offset],
                             [self.tree[x_coo][y].data, self.tree[x_coo + 1][y + 1].data], "k-")
        return temp_ax


def calculate_future_price(price: float, proba_up: float, proba_down: float, iter_1: int, iter_2: int) -> float:
    # formula for value at i-th step s_i_j = S_0 * u^i * d^(i-j)
    return price * (proba_up ** (iter_1 - iter_2)) * (proba_down ** iter_2)


def calculate_parameters_binomial(price_window: npt.NDArray, step_size: int = 1, r: float = 0) -> tuple[
        float, float, int, float]:
    up_jump = np.exp((price_window.std().iloc[0]) * np.sqrt(step_size / 10))
    down_jump = 1 / up_jump
    proba_up = (np.e ** (r * step_size) - down_jump) / (up_jump - down_jump)
    proba_down = 1 - proba_up
    return up_jump, down_jump, proba_up, proba_down


def create_feature_label_set(price_df: pd.DataFrame, window_length: int = 10) -> list[float]:
    temp_list = []
    for s in range(price_df.shape[0] - window_length):
        temp_window = price_df.iloc[s: s + window_length].reset_index()
        temp_window = temp_window.T
        temp_window.drop("index", axis=0, inplace=True)
        temp_window[window_length] = temp_window[window_length - 1]
        temp_window[window_length + 1] = price_df.iloc[s + window_length]
        temp_list.append(temp_window)
    return temp_list


def binomial_prediction_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # the price at s_t we are trying to estimate is equal. s_t = pSu + (1-q)Sd, u and d we get from the above function
    s_t = y_true[:, 0] * y_pred[:, 0] * u + y_true[:, 0] * y_pred[:, 1] * d
    return tf.reduce_mean(tf.square(y_true[:, 1] - s_t))


def neg_binomial_prediction_error(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    # the price at s_t we are trying to estimate is equal. s_t = pSu + (1-q)Sd, u and d we get from the above function
    s_t = y_true[:, 0] * y_pred[:, 0] * u + y_true[:, 0] * y_pred[:, 1] * d
    return -1 * (tf.reduce_mean(tf.square(y_true[:, 1] - s_t))).numpy()


def build_model(hidden_layers: int = 1, no_of_neurons: int = 100, lr: float = 25e-3, drop_out: float = .20,
                recurrent_drop_out: float = .20):
    model = keras.models.Sequential()
    for layer in range(1, hidden_layers):
        model.add(keras.layers.LSTM(no_of_neurons, dropout=drop_out, recurrent_dropout=recurrent_drop_out,
                                    return_sequences=True))
    if hidden_layers > 0:
        model.add(keras.layers.LSTM(no_of_neurons, dropout=0.2, recurrent_dropout=0.2))
    model.add(keras.layers.Dense(2, activation="softmax"))
    model.compile(loss=binomial_prediction_error, optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model


def make_batches(data_x: list, data_y: list, size_of_batch: int) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    to_batch_x = tf.data.Dataset.from_tensor_slices(data_x)
    to_batch_y = tf.data.Dataset.from_tensor_slices(data_y)
    batched_x = to_batch_x.shuffle(buffer_size=100).batch(size_of_batch, drop_remainder=True).prefetch(1)
    batched_y = to_batch_y.shuffle(buffer_size=100).batch(size_of_batch, drop_remainder=True).prefetch(1)
    return batched_x, batched_y


def standard_scale(data, mean, deviation):
    return (data - mean) / deviation


# ensures reproducibility
np.random.seed(42)
tf.random.set_seed(42)

random_exponential_normal = np.exp(np.random.randn(250))  # 250 samples of the normal distribution with mean=0 and std=1

exponential_normal_mean = random_exponential_normal.mean()
exponential_normal_std = random_exponential_normal.std()

x = np.linspace(np.min(random_exponential_normal), np.max(random_exponential_normal), 250)

plt.plot(x, random_exponential_normal, color="black", label="exponential random normal variable")
plt.axhline(y=exponential_normal_mean)

for scalar, color in enumerate(("green", "blue", "red")):
    plt.axhline(y=(scalar + 1) * exponential_normal_std, color=color, label=f"{scalar + 1} * std. deviation", alpha=.4)
    plt.fill_between(x, scalar * exponential_normal_std, (scalar + 1) * exponential_normal_std, color=color, alpha=0.2)

plt.grid()
plt.legend()
plt.show()

WINDOW_SIZE = 10

nvidia_data = pd.read_csv(os.curdir + r"\NVDA.csv")
nvidia_prices = nvidia_data.loc[:, "Close"]
rolling_prices = nvidia_prices.rolling(WINDOW_SIZE)
rolling_prices_mean, rolling_prices_std = rolling_prices.mean(), rolling_prices.std()

rolling_sigma_prices_upper = rolling_prices_mean + rolling_prices_std
rolling_sigma_prices_lower = rolling_prices_mean - rolling_prices_std

for prices, labels in zip(
        (nvidia_prices[WINDOW_SIZE - 1:], rolling_sigma_prices_upper, rolling_sigma_prices_lower, rolling_prices_mean),
        ("Actual Prices", "Mean + Std.Dev", "Mean - Std.Dev", "Mean")):
    prices.plot(label=labels)

plt.legend()
plt.grid()
plt.show()

stdrdized_prices = (nvidia_prices - nvidia_prices.mean()) / nvidia_prices.std()

train_set_prices = nvidia_prices[:101]
prices_t_1 = train_set_prices.iloc[:-1].reset_index()
prices_t_0 = train_set_prices.iloc[1:].reset_index()
prices_t_0.drop("index", axis="columns", inplace=True)
prices_t_1.drop("index", axis="columns", inplace=True)

p = log_returns = prices_t_1 / prices_t_0

plt.hist(p, density=True, bins=15, ec="black")
plt.xlabel("Log returns")
plt.ylabel("Magnitude of the returns")
plt.show()

for_q_plot = np.array(p.iloc[:])
fig = plt.figure()
ax = fig.add_subplot(111)
stats.probplot(for_q_plot.ravel(), sparams=(p.std().iloc[0]), dist=lognorm, plot=ax)
ax.set_title(f"Q-Q Plot for a log-normal pdf, with the standard deviation of: {p.std().iloc[0]:.2f}")
plt.show()

u, d, p_u, p_d = calculate_parameters_binomial(log_returns, WINDOW_SIZE)

labeled_set = np.array(create_feature_label_set(nvidia_prices, WINDOW_SIZE))
labeled_set = labeled_set.reshape(-1, WINDOW_SIZE + 2)

# dummy_y variable is needed so that the train_test_split can work with RNN models
dummy_y = np.zeros(labeled_set.shape)
x_train_full, x_test, *_ = train_test_split(labeled_set, dummy_y, train_size=0.8, test_size=0.2)
dummy_y = np.zeros(x_train_full.shape)

x_train, x_val, *_ = train_test_split(x_train_full, dummy_y, train_size=0.8, test_size=0.2)
x_windows, y_prices = x_train[:, :-2], x_train[:, -2:]
x_val_windows, y_val_prices = x_val[:, :-2], x_val[:, -2:]

"""
Example of calculating the loss using a class instead of a fn
class BinomialTreeLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        s_t = (y_pred[:, 0] * y_true[0][0] * u + y_true[0][0] * y_pred[:, 1] * d)
        return s_t
"""

x_windows = x_windows.reshape(-1, WINDOW_SIZE, 1)  # reshaped for as a seq for RNN models
kerasreg_cv = keras.wrappers.scikit_learn.KerasRegressor(build_model)

param_distribs = {
    "hidden_layers": randint(2, 10),
    "no_of_neurons": randint(10, 500),
    "lr": reciprocal(3e-4, 3e-2),
    "drop_out": reciprocal(2e-1, 5e-1),
    "recurrent_drop_out": reciprocal(2e-1, 5e-1),
}

neg_bin_scorrer = make_scorer(neg_binomial_prediction_error, greater_is_better=True)

scale_mean, scale_deviation = np.mean(x_windows), np.std(x_windows)

x_windows_train, y_prices, x_val_windows, y_val_prices = map(
    lambda dataset: standard_scale(np.log(dataset), scale_mean, scale_deviation),
    [x_windows, y_prices, x_val_windows, y_val_prices])

rnd_search_cv = RandomizedSearchCV(kerasreg_cv, param_distributions=param_distribs, n_iter=1, cv=2,
                                   scoring=neg_bin_scorrer)
rnd_search_cv.fit(x_windows_train, y_prices, epochs=5, validation_data=(x_val_windows, y_val_prices),
                  batch_size=1, verbose=2, callbacks=[keras.callbacks.EarlyStopping(patience=50),
                                                      keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.9,
                                                                                        patience=20)])
print(f"Best parameters: {rnd_search_cv.best_params_},\n best_score: {rnd_search_cv.best_score_}")

bin_model = rnd_search_cv.best_estimator_.model
print(bin_model.summary())

print("model probabilities:")
predicted_probabilities = bin_model.predict(x_test[:, :-2, np.newaxis])

np.set_printoptions(suppress=True)
print(predicted_probabilities)
"""
bin_model.fit(x_windows_train, y_prices, epochs=100, validation_data=(x_val_windows, y_val_prices), batch_size=1,
              verbose=1, callbacks=[keras.callbacks.EarlyStopping(patience=100),
                                    keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.01, patience=80)])
"""
x_test_model = np.log(x_test)
x_test_model = standard_scale(x_test_model, scale_mean, scale_deviation)

price_probas_bin_ml = bin_model.predict(x_test_model[:, :-2, np.newaxis])
predicted_prices_ml = price_probas_bin_ml * x_test[:, -2:-1] * [u, d]
predicted_prices_ml = [np.sum(x) for x in predicted_prices_ml]

predicted_prices_classic_fixed_probabilities = x_test[:, -2:-1] * [u, d] * [p_u, p_d]
predicted_prices_classic_fixed_probabilities = [np.sum(x) for x in predicted_prices_classic_fixed_probabilities]


def calculate_mse(y_true, y_pred):
    return np.mean(np.square(y_true - y_pred))


dict_all = {"actual prices": x_test[:, -1:],
            "new model probabilities": np.array(predicted_prices_ml),
            "old model probabilities": np.array(predicted_prices_classic_fixed_probabilities),
            }

subplot_no = 411
all_labels = tuple([*dict_all.keys()])

for combination in [all_labels] + list(itertools.combinations(all_labels, 2)):
    ax = plt.subplot(subplot_no)
    temp_data = []
    for label in combination:
        ax.plot(dict_all[label], label=label)
        if len(combination) < 3:
            temp_data.append(dict_all[label])
    title = " vs. ".join(combination)
    if len(temp_data) > 0:
        current_MSE = calculate_mse(*temp_data)
        title = title + f" MSE: {current_MSE:.4f}"
    ax.set_title(title)
    plt.grid()
    plt.legend(bbox_to_anchor=(1.25, 1), loc="upper right")
    subplot_no += 1

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.789,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)
plt.show()

all_probs = bin_model.predict(x_test[:1, :-2, np.newaxis])
print("all probabilities:", all_probs)

probability_up = all_probs[0][0]
probability_down = all_probs[0][1]

print(f"{probability_up=}, {probability_down=}")
price_t0 = x_test[:, -2:-1]

print("Price at time zero", price_t0)
predictedProbabilityPrice = BinomialPriceTree(price_t0[0][0], u, d, probability_up, probability_down,
                                              calculate_future_price)
predictedProbabilityPrice.calculate_prices()
predictedProbabilityPrice.plot_tree()
predictedProbabilityPrice.print_tree(return_matrix=False)
print("Expected price: ", predictedProbabilityPrice.calculate_expected_price())

binTree = BinomialPriceTree(nvidia_prices.iloc[WINDOW_SIZE], u, d, probability_up, probability_down,
                            calculate_future_price)
binTree.calculate_prices()

off_tree = binTree.get_plot(WINDOW_SIZE)
off_tree.plot(nvidia_prices.iloc[:20])
plt.show()
