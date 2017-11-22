import matplotlib.pyplot as plt

class ReconstructionPlotter:

    def __init__(self, num_reconstructions, file_name_base):
        self.num_reconstructions = num_reconstructions 
        self.file_name_base = file_name_base


    def plotReconstruction(self, original, reconstructed, name=""): 
        for i in range(self.num_reconstructions):
            # original
            ax = plt.subplot(2, self.num_reconstructions, i+1)
            plt.imshow(original[i].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # reconstructed
            ax = plt.subplot(2, self.num_reconstructions, i+1+self.num_reconstructions)
            plt.imshow(reconstructed[i].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

        plt.savefig(self.file_name_base + name + '.png')

    def plotTwoPassWithDiff(self, original, first_pass, second_pass, name=""): 
        plt.figure(figsize=(10,4))        

        for i in range(self.num_reconstructions):
            # original
            ax = plt.subplot(4, self.num_reconstructions, i+1)
            plt.imshow(original[i].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # reconstructed
            ax = plt.subplot(4, self.num_reconstructions, i+1+self.num_reconstructions)
            plt.imshow(first_pass[i].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(4, self.num_reconstructions, i+1+2*self.num_reconstructions)
            plt.imshow(second_pass[i].reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(4, self.num_reconstructions, i+1+3*self.num_reconstructions)
            diff = abs(second_pass[i]-first_pass[i])
            plt.imshow(diff.reshape(28,28))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)


        plt.subplots_adjust(wspace=0.2, hspace=0)
        plt.savefig(self.file_name_base + name + '.png')