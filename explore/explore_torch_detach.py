import torch


def main():
    x = torch.rand(size=(10, 3))
    y = torch.rand(size=(10, 3))
    x.requires_grad = True
    y.requires_grad = True
    z = x + y
    z += x * y
    print(z.requires_grad)
    print(z.grad_fn.next_functions)

    z_detached = z.detach().cpu()
    print(z_detached.requires_grad)
    print(z_detached.grad_fn)
    print(z.requires_grad)
    print(z.grad_fn.next_functions)


if __name__ == "__main__":
    main()
