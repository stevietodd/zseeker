#include <iostream>
//#include <numeric>
#include <limits>
using namespace std;

int main(int argc, char *argv[])
{
	typedef numeric_limits<char> limchar;
	typedef numeric_limits<int> limint;
	typedef numeric_limits<float> limfloat;
	typedef numeric_limits<double> limdouble;
	typedef numeric_limits<long double> limlongdouble;

	cout << "Size of char: " << sizeof(char) << " byte(s)" << "\n";
	cout << "Digits of char: " << limchar::digits << " digit(s)" << "\n";
	cout << "Digits 10 of char: " << limchar::digits10 << " digit(s)" << "\n";
	cout << "Max Digits 10 of char: " << limchar::max_digits10 << " digit(s)" << "\n";
	cout << "------------------" << "\n";

	cout << "Size of int: " << sizeof(int) << " byte(s)" << "\n";
	cout << "Digits of int: " << limint::digits << " digit(s)" << "\n";
	cout << "Digits 10 of int: " << limint::digits10 << " digit(s)" << "\n";
	cout << "Max Digits 10 of int: " << limint::max_digits10 << " digit(s)" << "\n";
	cout << "------------------" << "\n";

	cout << "Size of float: " << sizeof(float) << " byte(s)" << "\n";
	cout << "Digits of float: " << limfloat::digits << " digit(s)" << "\n";
	cout << "Digits 10 of float: " << limfloat::digits10 << " digit(s)" << "\n";
	cout << "Max Digits 10 of float: " << limfloat::max_digits10 << " digit(s)" << "\n";
	cout << "------------------" << "\n";

	cout << "Size of double: " << sizeof(double) << " byte(s)" << "\n";
	cout << "Digits of double: " << limdouble::digits << " digit(s)" << "\n";
	cout << "Digits 10 of double: " << limdouble::digits10 << " digit(s)" << "\n";
	cout << "Max Digits 10 of double: " << limdouble::max_digits10 << " digit(s)" << "\n";
	cout << "------------------" << "\n";

	cout << "Size of long double: " << sizeof(long double) << " byte(s)" << "\n";
	cout << "Digits of long double: " << limlongdouble::digits << " digit(s)" << "\n";
	cout << "Digits 10 of long double: " << limlongdouble::digits10 << " digit(s)" << "\n";
	cout << "Max Digits 10 of long double: " << limlongdouble::max_digits10 << " digit(s)" << "\n";
	cout << "------------------" << "\n";

	return 0;
}