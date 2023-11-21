// #include <python3.11/moduleobject.h>
#define PY_SSIZE_T_CLEAN
#include <Python.h>
// #include <numpy/arrayobject.h>
// #include <python3.11/Python.h>
#include <arrayobject.h>
#include <cmath>

#include "gaussian3d.hpp"
#include "matrices.hpp"


static PyObject *compute_overlap(PyObject *self,
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

static PyObject *compute_kinetic(PyObject *self,
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

static PyObject *compute_nuclear(PyObject *self,
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

static PyObject *compute_two_electron_integrals(
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

static PyMethodDef methods[] = {
    {.ml_name="compute_overlap",
     .ml_meth=compute_overlap,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the overlap matrix.",
    },
    {.ml_name="compute_kinetic",
     .ml_meth=compute_kinetic,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the kinetic energy matrix.",
    },
    {.ml_name="compute_nuclear",
     .ml_meth=compute_nuclear,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the potential energy matrix.",
    },
    {.ml_name="compute_two_electron_integrals",
     .ml_meth=compute_two_electron_integrals,
     .ml_flags=METH_VARARGS,
     .ml_doc="Compute the two electron integrals tensor.",
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
