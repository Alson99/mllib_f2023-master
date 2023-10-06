from enum import IntEnum

SetType = IntEnum('SetType',('train','valid','test'))
TrainType = IntEnum('TrainType',('Train','gradient_descent','normal_equation'))
