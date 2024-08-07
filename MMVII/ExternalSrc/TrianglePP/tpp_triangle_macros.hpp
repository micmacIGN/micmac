 /** 
    @file  tpp_triangle_macros.hpp    
    @brief Helper macros used to acces TriLib's internal data 
 */

#ifndef TRPP_TRILIB_MACROS
#define TRPP_TRILIB_MACROS


// Macros to hide the most common casts from void*
//  -- OPEN TODO:: remove, use typed pointers instead (needs a bigger refactoring!)

#define TP_MESH_BEHAVIOR() \
    Triwrap::__pmesh* tpmesh = static_cast<Triwrap::__pmesh *>(m_pmesh); \
    Triwrap::__pbehavior* tpbehavior = static_cast<Triwrap::__pbehavior *>(m_pbehavior);

#define TP_MESH() \
    Triwrap::__pmesh* tpmesh = static_cast<Triwrap::__pmesh *>(m_pmesh);

#define TP_MESH_BEHAVIOR_WRAP() \
    Triwrap::__pmesh* tpmesh = static_cast<Triwrap::__pmesh *>(m_pmesh); \
    Triwrap::__pbehavior* tpbehavior = static_cast<Triwrap::__pbehavior *>(m_pbehavior); \
    Triwrap* pTriangleWrap = static_cast<Triwrap *>(m_triangleWrap);

#define TP_WRAP_PTR() \
    static_cast<Triwrap *>(m_triangleWrap);

#define TP_MESH_PTR() \
    static_cast<Triwrap::__pmesh *>(m_pmesh)

#define TP_BEHAVIOR_PTR() \
    static_cast<Triwrap::__pbehavior *>(m_pbehavior)

#define TP_INPUT() \
   triangulateio* pin = static_cast<triangulateio*>(m_in);

#define TP_VOROUT() \
    triangulateio* tpvorout = static_cast<triangulateio*>(m_vorout);


// OPEN TODO:: replace old-style casts with modern ones!!!
//  --> in the macros below:

#define TP_PLOOP_PTR(fit) ((Triwrap::__otriangle *)(&((fit).floop)))

#define TP_MESH_PLOOP(fit) \
     Triwrap::__pmesh  * tpmesh  = (Triwrap::__pmesh *) (fit.m_delaunay->m_pmesh); \
     Triwrap::__otriangle* ploop = (Triwrap::__otriangle*)(&(fit.floop));

#define TP_MESH_WRAP_ITER() \
    Triwrap::__pmesh* tpmesh = static_cast<Triwrap::__pmesh *>(m_delaunay->m_pmesh); \
    Triwrap* pTriangleWrap = static_cast<Triwrap *>(m_delaunay->m_triangleWrap);

#define TP_BEHAVIOR_ITER() \
    Triwrap::__pbehavior* tpbehavior = static_cast<Triwrap::__pbehavior *>(m_delaunay->m_pbehavior);

#define TP_MESH_ITER() \
    Triwrap::__pmesh* tpmesh = static_cast<Triwrap::__pmesh *>(m_delaunay->m_pmesh);

#define TP_PLOOP_ITER() \
    Triwrap::__otriangle* ploop = (Triwrap::__otriangle*)(&(this->floop));

#endif // TRPP_TRILIB_MACROS
