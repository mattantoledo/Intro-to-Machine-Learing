#################################
# Your name: Mattan Toledo
#################################

import numpy as np
import matplotlib.pyplot as plt
import intervals


class Assignment2(object):
    """Assignment 2 skeleton.

    Please use these function signatures for this assignment and submit this file, together with the intervals.py.
    """

    def sample_from_D(self, m):
        """Sample m data samples from D.
        Input: m - an integer, the size of the data sample.

        Returns: np.ndarray of shape (m,2) :
                A two dimensional array of size m that contains the pairs where drawn from the distribution P.
        """
        x = np.random.uniform(low=0.0, high=1.0, size=m)
        x = np.sort(x)
        cond = (((x >= 0) & (x <= 0.2)) | ((x >= 0.4) & (x <= 0.6)) | ((x >= 0.8) & (x <= 1)))
        prob_label_1 = np.array([0.8 if c else 0.1 for c in cond])
        prob_label_0 = 1 - prob_label_1
        prob = np.transpose([prob_label_0, prob_label_1])
        y = [np.random.choice(a=[0, 1], p=p) for p in prob]

        res = np.transpose([x, y])
        return res

    def experiment_m_range_erm(self, m_first, m_last, step, k, T):
        """Runs the ERM algorithm.
        Calculates the empirical error and the true error.
        Plots the average empirical and true errors.
        Input: m_first - an integer, the smallest size of the data sample in the range.
               m_last - an integer, the largest size of the data sample in the range.
               step - an integer, the difference between the size of m in each loop.
               k - an integer, the maximum number of intervals.
               T - an integer, the number of times the experiment is performed.

        Returns: np.ndarray of shape (n_steps,2).
            A two dimensional array that contains the average empirical error
            and the average true error for each m in the range accordingly.
        """

        m_vec = []
        emp_avg_vec = []
        true_avg_vec = []

        for m in range(m_first, m_last + 1, step):
            #print("m = " + str(m) + " ")
            emp_lst = []
            true_lst = []

            for t in range(T):

                sample = self.sample_from_D(m)
                xs = sample[:, 0]
                ys = sample[:, 1]

                res_intervals, best_error = intervals.find_best_interval(xs, ys, k)

                empiric_error = best_error / m
                true_error = self.calculate_true_error(res_intervals)

                emp_lst.append(empiric_error)
                true_lst.append(true_error)

            m_vec.append(m)
            emp_avg_vec.append(np.mean(emp_lst))
            true_avg_vec.append(np.mean(true_lst))

        plt.plot(m_vec, emp_avg_vec, label="averaged empiric")
        plt.plot(m_vec, true_avg_vec, label="averaged true")
        plt.title('(b) Averaged Empiric and True Errors')
        plt.xlabel('n')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
        return

    def experiment_k_range_erm(self, m, k_first, k_last, step):
        """Finds the best hypothesis for k= 1,2,...,10.
        Plots the empirical and true errors as a function of k.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the ERM algorithm.
        """
        k_vec = []
        emp_vec = []
        true_vec = []

        sample = self.sample_from_D(m)
        xs = sample[:, 0]
        ys = sample[:, 1]

        for k in range(k_first, k_last + 1, step):
            #print("k = " + str(k) + " ")
            res_intervals, best_error = intervals.find_best_interval(xs, ys, k)

            empiric_error = best_error / m
            true_error = self.calculate_true_error(res_intervals)

            k_vec.append(k)
            emp_vec.append(empiric_error)
            true_vec.append(true_error)

        plt.plot(k_vec, emp_vec, label="empiric error")
        plt.plot(k_vec, true_vec, label="true error")
        plt.title('(c) Empiric and True Errors as a function of k')
        plt.xlabel('k')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
        return

    def experiment_k_range_srm(self, m, k_first, k_last, step):
        """Run the experiment in (c).
        Plots additionally the penalty for the best ERM hypothesis.
        and the sum of penalty and empirical error.
        Input: m - an integer, the size of the data sample.
               k_first - an integer, the maximum number of intervals in the first experiment.
               m_last - an integer, the maximum number of intervals in the last experiment.
               step - an integer, the difference between the size of k in each experiment.

        Returns: The best k value (an integer) according to the SRM algorithm.
        """
        k_vec = []
        emp_vec = []
        true_vec = []
        penalty_vec = []

        sample = self.sample_from_D(m)
        xs = sample[:, 0]
        ys = sample[:, 1]

        for k in range(k_first, k_last + 1, step):
            #print("k = " + str(k) + " ")
            res_intervals, best_error = intervals.find_best_interval(xs, ys, k)

            empiric_error = best_error / m
            true_error = self.calculate_true_error(res_intervals)

            k_vec.append(k)
            emp_vec.append(empiric_error)
            true_vec.append(true_error)
            penalty_vec.append(self.calc_penalty(k, m))

        emp_penalty_vec = np.array(emp_vec) + np.array(penalty_vec)
        plt.plot(k_vec, emp_vec, label="empiric error")
        plt.plot(k_vec, true_vec, label="true error")
        plt.plot(k_vec, penalty_vec, label="penalty")
        plt.plot(k_vec, emp_penalty_vec, label="empiric error + penalty")
        plt.title('(d) Empiric and True Errors with penalty as a function of k')
        plt.xlabel('k')
        plt.legend()
        plt.show()
        return

    def cross_validation(self, m):
        """Finds a k that gives a good test error.
        Input: m - an integer, the size of the data sample.

        Returns: The best k value (an integer) found by the cross validation algorithm.
        """

        sample = self.sample_from_D(m)
        np.random.shuffle(sample)

        sample_sorted = sorted(sample[:int(0.8*m)], key=lambda x:x[0])
        train_data = np.array(sample_sorted)
        validation_data = sample[int(0.8*m):]

        xt = train_data[:, 0]
        yt = train_data[:, 1]
        xv = validation_data[:, 0]
        yv = validation_data[:, 1]

        best_k_intervals = []
        k_vec = []
        for k in range(1, 11, 1):
            #print("k = " + str(k) + " ")
            res_intervals, _ = intervals.find_best_interval(xt, yt, k)
            best_k_intervals.append(res_intervals)
            k_vec.append(k)

        validation_error_vec = [self.calc_empiric_error(h, xv, yv) for h in best_k_intervals]
        validation_error_vec = np.array(validation_error_vec)

        min_err = np.min(validation_error_vec)
        min_index = np.argmin(validation_error_vec)
        best_k = k_vec[min_index]

        print("(e)")
        print("minimum error = " + str(min_err))
        print("minimum error best k = " + str(best_k))
        print("best intervals are:")
        print(best_k_intervals[min_index])
        return


    #################################
    # Place for additional methods

    # calculates all the intervals in [0,1] which are not the given intervals
    def calc_negative_intervals(self, intervals):
        neg_intervals = []
        curr = (0.0, intervals[0][0])
        neg_intervals.append(curr)
        n = len(intervals)
        for i in range(n-1):
            curr = (intervals[i][1], intervals[i+1][0])
            neg_intervals.append(curr)
        curr = (intervals[n-1][1], 1.0)
        neg_intervals.append(curr)
        return neg_intervals

    # calculate the length of the intersection between two intervals
    def intersect_intervals(self, i1, i2):
        l1, u1 = i1
        l2, u2 = i2

        # no intersection
        if u1 <= l2 or u2 <= l1:
            return 0

        # i2 is a subset of i1
        if l1 <= l2 <= u2 <= u1:
            return u2 - l2

        # i1 is a subset of i2
        if l2 <= l1 <= u1 <= u2:
            return u1 - l1

        # real intersection option 1
        if l1 <= l2 <= u1 <= u2:
            return u1 - l2

        # real intersection option 2
        if l2 <= l1 <= u2 <= u1:
            return u2 - l1

    def calculate_true_error(self, h_intervals1):

        h_intervals0 = self.calc_negative_intervals(h_intervals1)
        p_intervals1 = [(0.0, 0.2), (0.4, 0.6), (0.8, 1.0)]
        p_intervals0 = self.calc_negative_intervals(p_intervals1)

        p1_intervals1 = 0.8 # P[y=1|x] = 0.8
        p1_intervals0 = 0.1 # P[y=1|x] = 0.1

        p0_intervals1 = 1 - p1_intervals1 # P[y=0|x] = 0.2
        p0_intervals0 = 1 - p1_intervals0 # P[y=0|x] = 0.9
        e = 0

        # using law of total expectation on each interval where x can be
        # error is a case when h(x) != y
        for h_i1 in h_intervals1:  # h(x) = 1
            for p_i1 in p_intervals1:
                e += self.intersect_intervals(h_i1, p_i1) * p0_intervals1
            for p_i0 in p_intervals0:
                e += self.intersect_intervals(h_i1, p_i0) * p0_intervals0

        for h_i0 in h_intervals0:  # h(x) = 0
            for p_i1 in p_intervals1:
                e += self.intersect_intervals(h_i0, p_i1) * p1_intervals1
            for p_i0 in p_intervals0:
                e += self.intersect_intervals(h_i0, p_i0) * p1_intervals0

        return e

    # calculates the penalty for SRM
    def calc_penalty(self, k, n):
        vc_dim_k = 2 * k
        delta = 0.1
        return 2 * np.sqrt((vc_dim_k + np.log(2/delta)) / n)

    def calc_empiric_error(self, predictor, xs, ys):

        err = 0
        for i in range(len(xs)):
            prediction = 0

            for curr_interval in predictor:
                curr_l = curr_interval[0]
                curr_u = curr_interval[1]

                if curr_l <= xs[i] <= curr_u:
                    prediction = 1
                    break

            err += (prediction != ys[i])

        return err / len(xs)
    #################################


if __name__ == '__main__':

    ass = Assignment2()
    ass.experiment_m_range_erm(10, 100, 5, 3, 100)
    ass.experiment_k_range_erm(1500, 1, 10, 1)
    ass.experiment_k_range_srm(1500, 1, 10, 1)
    ass.cross_validation(1500)

    print('done')

