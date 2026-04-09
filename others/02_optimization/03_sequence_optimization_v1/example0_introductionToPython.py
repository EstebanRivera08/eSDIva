import matplotlib.pyplot as plt
import numpy as np

nx = 100
x = np.linspace(-np.pi, np.pi, nx)


def complex_wave(x, omega=1):
    "from x create exp(-j*x)=cos(x)+j*sin(x)"
    return np.exp(-1j * omega * x)


# Let's check the function
y = complex_wave(x)

print("y is a complex array of shape", y.shape)
print("The first 5 values of y are:", y[:5])

# Now let's create a class to take and store the complex wave


class ComplexWave:
    def __init__(self, x, omega=1):
        self.set_params(x, omega)

    def get_magnitude(self):
        return np.abs(self.y)

    def get_phase(self):
        return np.angle(self.y)

    def set_params(self, x, omega=1):
        self.x = x
        self.omega = omega
        self.y = complex_wave(x, omega)
        self.magnitude = self.get_magnitude()
        self.phase = self.get_phase()
        self.real_part = self.y.real
        self.imaginary_part = self.y.imag

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.x, self.real_part, label="Real Part")
        plt.plot(self.x, self.imaginary_part, label="Imaginary Part")
        plt.title("Complex Wave: exp(-j*x)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    def __call__(self, x, omega):
        self.set_params(x, omega)

    def __repr__(self):
        return f"ComplexWave(x=[{self.x[0]}-{self.x[-1]}], omega={self.omega})"


# Let's create an instance of the ComplexWave class and plot it
wave = ComplexWave(x)
print(wave)

# We can access the class attributes
print("omega:", wave.omega, "magnitude shape:", wave.magnitude.shape)

# We can also call the instance's plot method to visualize the wave
wave.plot()

# Finally, we can change the parameters of the wave and see how it affects the plot
wave.set_params(x, omega=2)

print("Updated omega:", wave.omega)
wave.plot()

# We can also use the __call__ method to update the parameters
wave(x, omega=3)

print("Updated omega using __call__:", wave.omega)
wave.plot()
