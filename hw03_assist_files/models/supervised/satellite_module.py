import torch
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import torchmetrics

from hw03_assist_files.models.supervised.segmentation_cnn import SegmentationCNN
from hw03_assist_files.models.supervised.unet import UNet
from hw03_assist_files.models.supervised.resnet_transfer import FCNResnetTransfer

class ESDSegmentation(pl.LightningModule):
    """
    LightningModule for training a segmentation model on the ESD dataset
    """
    def __init__(self, model_type, in_channels, out_channels, 
                 learning_rate=1e-3, model_params: dict = {}):
        """
        Initializes the model with the given parameters.

        Input:
        model_type (str): type of model to use, one of "SegmentationCNN",
        "UNet", or "FCNResnetTransfer"
        in_channels (int): number of input channels of the image of shape
        (batch, in_channels, width, height)
        out_channels (int): number of output channels of prediction, prediction
        is shape (batch, out_channels, width//scale_factor, height//scale_factor)
        learning_rate (float): learning rate of the optimizer
        model_params (dict): dictionary of parameters to pass to the model

        """
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate

        # get the model to use
        if model_type == "SegmentationCNN":
            self.model = SegmentationCNN(in_channels, out_channels, **model_params)
        elif model_type == "UNet":
            self.model = UNet(in_channels, out_channels, **model_params)
        elif model_type == "FCNResnetTransfer":
            self.model = FCNResnetTransfer(in_channels, out_channels, **model_params)

        # define performance metrics for segmentation task
        # such as accuracy per class accuracy, average IoU, per class IoU,
        # per class AUC, average AUC, per class F1 score, average F1 score
        # these metrics will be logged to weights and biases
            
        # FINISH METRICS LATER TODO AND FOCUS ON TRAINING AND VAL STEPS
        self.train_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task = 'multiclass', num_classes = 4),
            #TODO: Add more metrics if needed
            'weighted_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=4, average='weighted'),
            'IoU': torchmetrics.JaccardIndex(task="multiclass", num_classes=4),
            'average_IoU': torchmetrics.JaccardIndex(task="multiclass", num_classes=4, average='weighted'),
            'AUC': torchmetrics.AUROC(task='multiclass', num_classes=4),
            'average_AUC': torchmetrics.AUROC(task='multiclass', num_classes=4, average='weighted'),
            'f1_per_class' : torchmetrics.F1Score(task='multiclass',num_classes=4, average=None),
            'f1_score_weighted': torchmetrics.F1Score(task='multiclass',num_classes=4, average='weighted')
        })
        self.val_metrics = torchmetrics.MetricCollection({
            'accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=4),
            #TODO: Add more metrics if needed
            'weighted_accuracy': torchmetrics.Accuracy(task='multiclass', num_classes=4, average='weighted'),
            'IoU': torchmetrics.JaccardIndex(task="multiclass", num_classes=4),
            'average_IoU': torchmetrics.JaccardIndex(task="multiclass", num_classes=4, average='weighted'),
            'AUC': torchmetrics.AUROC(task='multiclass', num_classes=4),
            'average_AUC': torchmetrics.AUROC(task='multiclass', num_classes=4, average='weighted'),
            'f1_per_class' : torchmetrics.F1Score(task='multiclass',num_classes=4, average=None),
            'f1_score_weighted': torchmetrics.F1Score(task='multiclass',num_classes=4, average='weighted')
        })
       
    
    def forward(self, X):
        """
        Run the input X through the model

        Input: X, a (batch, input_channels, width, height) image
        Ouputs: y, a (batch, output_channels, width/scale_factor, height/scale_factor) image
        """
        return self.model(X)
    
    def training_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then uses CrossEntropyLoss to calculate
        the current loss.

        Note: CrossEntropyLoss requires mask to be of type
        torch.int64 and shape (batches, width, height), 
        it only has one channel as the label is encoded as
        an integer index. As these may not be this shape and
        type from the dataset, you might have to use
        torch.reshape or torch.squeeze in order to remove the
        extraneous dimensions, as well as using Tensor.to to
        cast the tensor to the correct type.

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            train_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        sat_img, mask, metadata = batch
        sat_img = sat_img.to(torch.float32)
        squeezed_mask = torch.squeeze(mask).to(torch.int64)
        train_prediction = self.forward(sat_img)
        # print(f"Train prediction shape: {train_prediction.shape}")
        # print(f"Train squeeze mask shape: {squeezed_mask.shape}")
        train_loss = nn.CrossEntropyLoss()(train_prediction, squeezed_mask)
        
        # TODO: log train_loss for W&B use
        self.log('Multi-Class Accuracy', self.train_metrics['accuracy'](train_prediction, squeezed_mask))
        self.log("Weighted Multi-Class Accuracy", self.train_metrics['weighted_accuracy'](train_prediction, squeezed_mask))
        self.log('Multi-Class AUC', self.train_metrics['AUC'](train_prediction, squeezed_mask))
        self.log('Average AUC', self.train_metrics['average_AUC'](train_prediction, squeezed_mask))
        self.log('Multi-Class IoU', self.train_metrics['IoU'](train_prediction, squeezed_mask))
        self.log('Average IoU', self.train_metrics['average_IoU'](train_prediction, squeezed_mask))
        self.log('F1 per class', self.train_metrics['f1_per_class'](train_prediction, squeezed_mask).mean())
        self.log('Average F1', self.train_metrics['f1_score_weighted'](train_prediction, squeezed_mask).mean())

        return train_loss
    
    
    def validation_step(self, batch, batch_idx):
        """
        Gets the current batch, which is a tuple of
        (sat_img, mask, metadata), predicts the value with
        self.forward, then evaluates the 

        Note: The type of the tensor input to the neural network
        must be the same as the weights of the neural network.
        Most often than not, the default is torch.float32, so
        if you haven't casted the data to be float32 in the
        dataset, do so before calling forward.

        Input:
            batch: tuple containing (sat_img, mask, metadata).
                sat_img: Batch of satellite images from the dataloader,
                of shape (batch, input_channels, width, height)
                mask: Batch of target labels from the dataloader,
                by default of shape (batch, 1, width, height)
                metadata: List[SubtileMetadata] of length batch containing 
                the metadata of each subtile in the batch. You may not
                need this.

            batch_idx: int indexing the current batch's index. You may
            not need this input, but it's part of the class' interface.

        Output:
            val_loss: torch.tensor of shape (,) (i.e, a scalar tensor).
            Should be the cross_entropy_loss, as it is the main validation
            loss that will be tracked.
            Gradients will not propagate unless the tensor is a scalar tensor.
        """
        # hazel: here we're basically doing the same as training_step but on the validation set?
        #        not sure tho

        sat_img, mask, metadata = batch
        sat_img = sat_img.to(torch.float32)
        squeezed_mask = torch.squeeze(mask).to(torch.int64)
        val_prediction = self.forward(sat_img)
        # print(f"Val prediction shape: {val_prediction.shape}")
        # print(f"Val squeeze mask shape: {squeezed_mask.shape}")
        val_loss = nn.CrossEntropyLoss()(val_prediction, squeezed_mask)
        # TODO: save val_loss for W&B use
        # self.log('val_loss', val_loss)
        self.log("Val Multi-Class Accuracy", self.val_metrics['accuracy'](val_prediction, squeezed_mask))
        self.log("Weighted Multi-Class Accuracy", self.val_metrics['weighted_accuracy'](val_prediction, squeezed_mask))
        self.log('Val Multi-Class AUC', self.val_metrics['AUC'](val_prediction, squeezed_mask))
        self.log('Average AUC', self.val_metrics['average_AUC'](val_prediction, squeezed_mask))
        self.log('Multi-Class IoU', self.val_metrics['IoU'](val_prediction, squeezed_mask))
        self.log('Average IoU', self.val_metrics['average_IoU'](val_prediction, squeezed_mask))
        self.log('F1 per class', self.val_metrics['f1_per_class'](val_prediction, squeezed_mask).mean())
        self.log('Average F1', self.val_metrics['f1_score_weighted'](val_prediction, squeezed_mask).mean())


        return val_loss
    
    def configure_optimizers(self):
        """
        Loads and configures the optimizer. See torch.optim.Adam
        for a default option.

        Outputs:
            optimizer: torch.optim.Optimizer
                Optimizer used to minimize the loss
        """
        return Adam(self.parameters(), lr=self.learning_rate)
