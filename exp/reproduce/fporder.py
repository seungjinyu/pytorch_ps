

class Testing:
    def __call__(self, *args, **kwds):
        
        return "testing inside"


t = Testing()

out = t()

print(out)

