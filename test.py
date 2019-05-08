import model

net=model.NetVLDAModel.NetVLADModelLF().cuda()
print(list(net.parameters()))
