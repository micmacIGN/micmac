 /** 
    @file  assert.hpp
    @brief Implements a better 'Assert'.
           Needed for the reviver::dpoint inplementation.
 */

namespace tpp {

#ifndef REVIVER_ASSERT_HPP
#define REVIVER_ASSERT_HPP

extern bool g_disableAsserts;
extern bool MyAssertFunction( bool b, const char* desc, int line, const char* file);

// macro
#if defined( _DEBUG )
#define Assert( exp, description ) tpp::g_disableAsserts \
          ? true \
          : tpp::MyAssertFunction( (int)(exp), description, __LINE__, __FILE__ )
#else
#define Assert( exp, description )
#endif

#endif

}
