import numpy as np
# Common loss class
class Loss:
    # Regularization loss calculation
    def regularaization_loss(self, layer):
        # 0 by default
        r_loss = 0

        # L1 regularization
        if layer.l1 > 0:
            r_loss += layer.l1 * np.sum(np.abs(layer.weights))
            r_loss += layer.l1 * np.sum(np.abs(layer.bias))
        # L2 regularization
        if layer.l2 > 0:
            r_loss += layer.l2 * np.sum(layer.weights * layer.weights)
            r_loss += layer.l2 * np.sum(layer.bias * layer.bias)
        return r_loss

    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y, *, include_regularization=False):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Add accumulated sum of losses and sample count
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    # Calculates accumulated loss
    def calculate_accumulated(self, *, include_regularization=False):

        # Calculate mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        # If just data loss - return it
        if not include_regularization:
            return data_loss

        # Return the data and regularization losses
        return data_loss, self.regularization_loss()

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0