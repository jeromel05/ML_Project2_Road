def unet(input_size = (80,80,3), nb_out_channels = 64):
    """Creates our implementation of UNet compiled with the following parameters:
        optimiser: Adam with default parameters
        loss: binary crossentropy
        metrics monitored: [f1_m, 'accuracy', precision_m, recall_m, nnztr_m, nnzte_m]

    Args:
        input_size: format of the input image
        nb_out_channels: number of output channels. Modifying this parameter increases the number of channel on each layer. It corresponds to the initial (after the first convolution) and output channel (before the last convolution) size  

    Returns:
        the precision of the model
    """
    inputs = Input(input_size)

    print(inputs.shape)
    conv1 = Conv2D(nb_out_channels, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(inputs)
    print(conv1.shape)
    drop1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(nb_out_channels, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop1)
    print(conv1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    print(pool1.shape)
    conv2 = Conv2D(nb_out_channels*2, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(pool1)
    print(conv2.shape)
    drop2 = Dropout(0.2)(conv2)                                                                                                        
    conv2 = Conv2D(nb_out_channels*2, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_out_channels*4, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(pool2)
    drop3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(nb_out_channels*4, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_out_channels*8, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(pool3)
    drop4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(nb_out_channels*8, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv6 = Conv2D(nb_out_channels*16, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(pool4)
    conv6 = Conv2D(nb_out_channels*16, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(conv6)
    #print(conv6.shape)

    up7 = Conv2D(nb_out_channels*8, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv4,up7], axis = 3)
    conv7 = Conv2D(nb_out_channels*8, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(merge7)
    drop7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(nb_out_channels*8, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop7)

    up8 = Conv2D(nb_out_channels*4, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv3,up8], axis = 3)
    conv8 = Conv2D(nb_out_channels*4, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(merge8)
    drop8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(nb_out_channels*4, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(drop8)

    up9 = Conv2D(nb_out_channels*2, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv2,up9], axis = 3)
    conv9 = Conv2D(nb_out_channels*2, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(merge9)
    drop9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(nb_out_channels*3, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(conv9)

    up10 = Conv2D(nb_out_channels, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(UpSampling2D(size = (2,2))(conv9))
    merge10 = concatenate([conv1,up10], axis = 3)
    conv10 = Conv2D(nb_out_channels, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(merge10)
    drop10 = Dropout(0.2)(conv10)
    conv10 = Conv2D(nb_out_channels, 3, activation = 'relu', padding = 'same', dilation_rate=2, kernel_initializer = 'glorot_uniform')(conv10)
    conv10 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'glorot_uniform')(conv10)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv10)
    print("ww", conv10.shape)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = [f1_m, 'accuracy', precision_m, recall_m, nnztr_m, nnzte_m])
    
    model.summary()

    return model