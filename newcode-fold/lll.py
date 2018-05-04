class CAnimal:
    def __init__(self, voice='hello'):  # voice初始化默认为hello
        self.voice = voice

    def Say(self):
        print(self.voice)


    def Run(self):
        pass  # 空操作语句（不做任何操作）


class CDog(CAnimal):  # 继承类CAnimal
    def SetVoice(self, voice):  # 子类增加函数SetVoice
        self.voice = voice

    def Run(self):  # 子类重载函数Run
        print('Running')

bobo = CDog()
bobo.SetVoice('My Name is BoBo!')  # 设置child.data为hello
bobo.Say()
bobo.Run()  