from amplpy import AMPL
from amplpy import Environment

def main():
    # Environment('full path to the AMPL installation directory')
    ampl = AMPL(Environment("C:\Program Files\\ampl"))
    
    ampl.set_option('solver', 'cplex')
    # Load the AMPL model from file
    ampl.read("10596841_10867595.mod")
    # Read data
    ampl.read_data("minimart-I-100.dat")

    ampl.solve()

    print('Objective function value: ', ampl.get_objective('obj').value())

    x = ampl.get_variable('x')

    print(x.getValues())

if __name__ == '__main__':
    main()