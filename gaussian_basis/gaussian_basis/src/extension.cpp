// #include <python3.12/moduleobject.h>
// #include <python3.12/methodobject.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
// #include <numpy/arrayobject.h>
// #include <python3.12/Python.h>
#include <arrayobject.h>
#include <cmath>

#include "gaussian3d.hpp"
#include "basis_function.hpp"
#include "matrices.hpp"


static PyObject *compute_overlap_from_primitives(PyObject *self,
                                                 PyObject *args) {
    PyObject *overlap = NULL;
    PyObject *gaussian = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &overlap,
                          &PyArray_Type, &gaussian)
        )
        return NULL;
    int n = PyArray_NBYTES(gaussian)/sizeof(Gaussian3D);
    set_overlap_elements((double *)PyArray_DATA(overlap), 
                         (Gaussian3D *)PyArray_DATA(gaussian),
                         n);
    return Py_None;
}

static PyObject *compute_overlap(PyObject *self, PyObject *args) {
    PyObject *overlap = NULL;
    PyObject *orbital_data = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &overlap,
                          &PyArray_Type, &orbital_data)
        )
        return NULL;
    set_overlap_elements((double *)PyArray_DATA(overlap),
                         (struct BasisFunction *)
                             ((long *)PyArray_DATA(orbital_data) + 1),
                         ((long *)PyArray_DATA(orbital_data))[0]
                         );
    return Py_None;
}

static PyObject *compute_kinetic_from_primitives(PyObject *self,
                                                 PyObject *args) {
    PyObject *kinetic = NULL;
    PyObject *gaussian = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &kinetic,
                          &PyArray_Type, &gaussian)
        )
        return NULL;
    int n = PyArray_NBYTES(gaussian)/sizeof(Gaussian3D);
    set_kinetic_elements((double *)PyArray_DATA(kinetic), 
                         (Gaussian3D *)PyArray_DATA(gaussian),
                         n);
    return Py_None;
}

static PyObject *compute_kinetic(PyObject *self, PyObject *args) {
    PyObject *kinetic = NULL;
    PyObject *orbital_data = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &kinetic,
                          &PyArray_Type, &orbital_data)
        )
        return NULL;      
    set_kinetic_elements((double *)PyArray_DATA(kinetic),
                         (struct BasisFunction *)
                             ((long *)PyArray_DATA(orbital_data) + 1),
                         ((long *)PyArray_DATA(orbital_data))[0]);
    return Py_None;
}

static PyObject *compute_nuclear_from_primitives(PyObject *self,
                                                 PyObject *args) {
    PyObject *nuclear = NULL;
    PyObject *gaussian = NULL;
    PyObject *nuc_loc = NULL;
    PyObject *charges= NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          &PyArray_Type, &nuclear,
                          &PyArray_Type, &nuc_loc,
                          &PyArray_Type, &charges,
                          &PyArray_Type, &gaussian)
        )
        return NULL;
    int n = PyArray_NBYTES(gaussian)/sizeof(Gaussian3D);
    int charge_count = PyArray_NBYTES(charges)/sizeof(int);
    set_nuclear_potential_elements((double *)PyArray_DATA(nuclear), 
                                   (Gaussian3D *)PyArray_DATA(gaussian),
                                   n, 
                                   (struct Vec3 *)PyArray_DATA(nuc_loc),
                                   (int *)PyArray_DATA(charges),
                                   charge_count);
    return Py_None;
}

static PyObject *compute_nuclear(PyObject *self,
                                 PyObject *args) {
    PyObject *nuclear = NULL;
    PyObject *orbital_data = NULL;
    PyObject *nuc_loc = NULL;
    PyObject *charges= NULL;
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
                          &PyArray_Type, &nuclear,
                          &PyArray_Type, &nuc_loc,
                          &PyArray_Type, &charges,
                          &PyArray_Type, &orbital_data)
        )
        return NULL;
    int charge_count = PyArray_NBYTES(charges)/sizeof(int);
    set_nuclear_potential_elements((double *)PyArray_DATA(nuclear), 
                                   (struct BasisFunction *)
                                    ((long *)PyArray_DATA(orbital_data) + 1),
                                   ((long *)PyArray_DATA(orbital_data))[0], 
                                   (struct Vec3 *)PyArray_DATA(nuc_loc),
                                   (int *)PyArray_DATA(charges),
                                   charge_count);
    return Py_None;
}

static PyObject *compute_two_electron_integrals_from_primitives(
    PyObject *self, PyObject *args
    ) {
    PyObject *two_electron_integrals = NULL;
    PyObject *gaussian = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &two_electron_integrals,
                          &PyArray_Type, &gaussian))
        return NULL;
    double *two_electrons_integrals_data
        = (double *)PyArray_DATA(two_electron_integrals);
    Gaussian3D *gaussian_data
        = (Gaussian3D *)PyArray_DATA(gaussian);
    int n = PyArray_NBYTES(gaussian)/sizeof(Gaussian3D);
    set_two_electron_integrals_elements(
        two_electrons_integrals_data,
        gaussian_data,
        n
    );
    return Py_None;
}

static PyObject *compute_two_electron_integrals(
    PyObject *self, PyObject *args
) {
    PyObject *two_electron_integrals = NULL;
    PyObject *orbital_data = NULL;
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &two_electron_integrals,
                          &PyArray_Type, &orbital_data))
        return NULL;
    set_two_electron_integrals_elements(
        (double *)PyArray_DATA(two_electron_integrals),
        (struct BasisFunction *)
           ((long *)PyArray_DATA(orbital_data) + 1),
        
        // Remember to use the PyArray_DATA macro!
        // I forgot to do this, costing many hours of debugging
        // trying to find what I did wrong.
        ((long *)PyArray_DATA(orbital_data))[0]
    );
    return Py_None;
}

static void print_data_contents(char *data) {
    long *arr_long = (long *)data;
    fprintf(stdout, "Number of basis functions: %d\n", arr_long[0]);
    BasisFunction *arr_basis_func = (BasisFunction *)(data + sizeof(long));
    for (int i = 0; i < arr_long[0]; i++) {
        fprintf(stdout, "Basis function index: %d\n", i);
        fprintf(stdout, "Number of primitives: %d\n",
                arr_basis_func[i].count);
        for (int j = 0; j < arr_basis_func[i].count; j++) {
            fprintf(stdout, "Gaussian primitive index %d:\n", j);
            fprintf(stdout, "amplitude: %g\n",
                arr_basis_func[i].primitives[j].amplitude());
            fprintf(stdout, "exponent: %g\n",
                arr_basis_func[i].primitives[j].orbital_exponent());
            fprintf(stdout, "angular 0: %g\n",
                arr_basis_func[i].primitives[j].angular()[0]);
            fprintf(stdout, "angular 1: %g\n",
                arr_basis_func[i].primitives[j].angular()[1]);
            fprintf(stdout, "angular 2: %g\n",
                arr_basis_func[i].primitives[j].angular()[2]);
            fprintf(stdout, "position 0: %g\n",
                arr_basis_func[i].primitives[j].position()[0]);
            fprintf(stdout, "position 1: %g\n",
                arr_basis_func[i].primitives[j].position()[1]);
            fprintf(stdout, "position 2: %g\n",
                arr_basis_func[i].primitives[j].position()[2]);
        }
    }
}

static PyObject *set_pointer_addresses(PyObject *self, PyObject *args) {
    PyObject *data = NULL;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &data))
        return NULL;
    char *arr = (char *)PyArray_DATA(data);
    long *arr_long = (long *)arr;
    int basis_func_count = arr_long[0];
    BasisFunction *arr_basis_func = (BasisFunction *)(arr + sizeof(long));
    Gaussian3D *arr_gaussian3d 
        = (Gaussian3D *)(arr + sizeof(long)
                         + basis_func_count*sizeof(BasisFunction));
    int index = 0;
    for (int i = 0; i < basis_func_count; i++) {
        arr_basis_func[i].primitives = &arr_gaussian3d[index];
        index += arr_basis_func[i].count;
    }
    print_data_contents((char *)arr);
    return Py_None;

}

static PyMethodDef methods[] = {
    {.ml_name="compute_overlap",
    .ml_meth=compute_overlap,
    .ml_flags=METH_VARARGS,
    .ml_doc="Compute the overlap matrix"
    },
    {.ml_name="compute_kinetic",
    .ml_meth=compute_kinetic,
    .ml_flags=METH_VARARGS,
    .ml_doc="Compute the kinetic energy matrix"
    },
    {.ml_name="compute_nuclear",
    .ml_meth=compute_nuclear,
    .ml_flags=METH_VARARGS,
    .ml_doc="Compute the potential energy matrix."
    },
    {.ml_name="compute_two_electron_integrals",
    .ml_meth=compute_two_electron_integrals,
    .ml_flags=METH_VARARGS,
    .ml_doc="Compute the two electron integrals tensor."
    },
    {.ml_name="compute_overlap_from_primitives",
     .ml_meth=compute_overlap_from_primitives,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the overlap matrix.",
    },
    {.ml_name="compute_kinetic_from_primitives",
     .ml_meth=compute_kinetic_from_primitives,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the kinetic energy matrix.",
    },
    {.ml_name="compute_nuclear_from_primitives",
     .ml_meth=compute_nuclear_from_primitives,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the potential energy matrix.",
    },
    {.ml_name="compute_two_electron_integrals_from_primitives",
     .ml_meth=compute_two_electron_integrals_from_primitives,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the two electron integrals tensor.",
    },
    {.ml_name="set_pointer_addresses",
     .ml_meth=set_pointer_addresses,
     .ml_flags=METH_VARARGS,
     .ml_doc="Set pointer addresses."
    },
    {.ml_name=NULL,
     .ml_meth=NULL,
     .ml_flags=0,
     .ml_doc="",
    },
};

static struct PyModuleDef module = {
    .m_base=PyModuleDef_HEAD_INIT,
    .m_name="module",
    .m_doc=NULL,
    .m_size=-1,
    .m_methods=methods,
};

PyMODINIT_FUNC PyInit_extension(void) {
    // https://stackoverflow.com/a/37944168
    import_array();
    return PyModule_Create(&module);
}
