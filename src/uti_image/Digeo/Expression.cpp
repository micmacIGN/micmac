#include "Expression.h"

#include <cstdlib>

using namespace std;

static const int DEFAULT_MIN_WIDTH = 0;

static const char
	OPEN_BRACES            = '{',
	CLOSE_BRACES           = '}',
	VARIABLE_TAG           = '$',
	SEPARATOR              = ':',
	DEFAULT_FILL_CHARACTER = '0';

#ifdef __DEBUG_EXPRESSION
	list<pair<Expression::Element*,string> > Expression::Element::__allocated_elements;
	list<Expression::Element::memory_transaction> Expression::Element::__memory_transactions;
#endif

//----------------------------------------------------------------------
// function related to class Expression
//----------------------------------------------------------------------

bool isInteger( const string &i_str )
{
	const size_t length = i_str.length();
	if ( length==0 ) return false;
	char c0 = i_str[0];
	if ( ( c0=='-' || c0=='+' ) && length<2 ) return false;
	else if ( c0<48 || c0>57 ) return false;
	for ( size_t i=0; i<length; i++ )
	{
		c0 = i_str[i];
		if ( c0<48 || c0>57 ) return false;
	}
	return true;
}

ostream & operator <<( ostream &io_stream, const Expression::Element &i_element )
{
	i_element.dump(io_stream);
	return io_stream;
}


//----------------------------------------------------------------------
// methods of class Expression::Element
//----------------------------------------------------------------------

bool Expression::Element::isVariable() const { return false; }


Expression::Element::~Element(){}

#ifdef __DEBUG_EXPRESSION
	void Expression::Element::__allocating( const std::string &i_msg )
	{
		__allocated_elements.push_back( pair<Element*,string>(this,i_msg) );

		__memory_transactions.push_back( memory_transaction(MT_NEW,this,i_msg) );
	}

	void Expression::Element::__deleting( const std::string &i_msg )
	{
		__memory_transactions.push_back( memory_transaction(MT_DELETE,this,i_msg) );

		list<pair<Element*,string> >::iterator it = __allocated_elements.begin();
		while ( it!=__allocated_elements.end() )
		{
			if ( it->first==this )
			{
				__allocated_elements.erase(it);
				return;
			}
			it++;
		}

		cerr << "DEBUG_ERROR: " << i_msg << ": deleting an object that has not been allocated (or is already deleted)" << endl;
		cerr << "object address : " << this << endl;
		__dump_allocated();
		__dump_transactions();

		cin.get();
		exit( EXIT_FAILURE );
	}

	void Expression::Element::__dump_allocated()
	{
		cerr << "undeleted objects :" << endl;
		list<pair<Element*,string> >::iterator it = __allocated_elements.begin();
		while ( it!=__allocated_elements.end() )
		{
			cerr << '\t' << it->first << " (" << it->second << ')' << endl;
			it++;
		}
	}

	void Expression::Element::__dump_transactions()
	{
		cerr << "transactions :" << endl;
		list<memory_transaction>::iterator it = __memory_transactions.begin();
		while ( it!=__memory_transactions.end() )
		{
			cerr << '\t';
			switch (it->type)
			{
			case MT_NEW: cerr<<"NEW"; break;
			case MT_DELETE: cerr<<"DELETE"; break;
			default: cerr<<"UNKNOWN";
			}
			cerr << ' ' << it->object << " (" << it->message << ')' << endl;
			it++;
		}
	}
#endif

Expression::Element * Expression::Element::duplicate( const Element &i_e )
{
	if ( i_e.isVariable() ) return new Expression::Variable( TO_CHILD(Expression::Variable,i_e) );
	return new Expression::String( TO_CHILD(Expression::String,i_e) );
}


//----------------------------------------------------------------------
// methods of class Expression::String
//----------------------------------------------------------------------

void Expression::String::dump( std::ostream &io_stream ) const { io_stream << "KEY_STR{" << mValue << '}'; }

string Expression::String::toString() const { return mValue; }

#ifdef __DEBUG_EXPRESSION
	Expression::String::String( const String &b ):
		Element(b)
	{
		__allocating("Expression::String::String(const String &)");
		mValue = b.mValue;
	}

	Expression::String::~String()
	{
		__deleting( "Expression::String::~String" );
	}
#endif


//----------------------------------------------------------------------
// methods of class Expression::Variable
//----------------------------------------------------------------------

Expression::Variable::Variable( string i_string ):
		mMinWidth(DEFAULT_MIN_WIDTH),
		mFillCharacter(DEFAULT_FILL_CHARACTER),
		mIsValid(false)
{
	#ifdef __DEBUG_EXPRESSION
		__allocating("Expression::Variable::Variable(string)");
	#endif

	mUnusedElements.push_back( i_string );
	if ( i_string.length()==0 ) return;

	// remove decoration if there's one
	if ( i_string[0]==VARIABLE_TAG )
	{
		if ( i_string.length()<4 || i_string[1]!=OPEN_BRACES || ( *i_string.rbegin() )!=CLOSE_BRACES ) return;
		i_string = i_string.substr( 2, i_string.length()-3 );
	}

	// find elements, separated by 'separator'
	list<string> elements;
	size_t pos0 = 0;
	for ( size_t i=0; i<i_string.length(); i++ )
	{
		if ( i_string[i]==SEPARATOR )
		{
			if ( i-pos0<1 ) return;
			elements.push_back( i_string.substr(pos0,i-pos0) );
			pos0 = i+1;
		}
	}
	if ( pos0==i_string.length() ) return;
	elements.push_back( i_string.substr(pos0) );

	// element : 0 = name (mandatory), 1 = minWidth, 2 = fillCharacter
	list<string>::iterator itElement = elements.begin();
	if ( itElement==elements.end() ) return;
	mName = *itElement++;
	if ( itElement!=elements.end() )
	{
		if ( !isInteger(*itElement) ) return;
		mMinWidth = atoi( (*itElement++).c_str() );
	}
	if ( itElement!=elements.end() )
	{
		if ( itElement->length()>1 ) return;
		mFillCharacter = (*itElement++)[0];
	}
	mIsValid = true;

	mUnusedElements.clear();
	while ( itElement!=elements.end() )
		mUnusedElements.push_back( *itElement++ );
}

string Expression::Variable::toString() const
{
	stringstream ss;
	ss << VARIABLE_TAG << OPEN_BRACES << mName;

	if ( mMinWidth!=DEFAULT_MIN_WIDTH || mFillCharacter!=DEFAULT_FILL_CHARACTER )
		ss << SEPARATOR << mMinWidth;
		
	if ( mFillCharacter!=DEFAULT_FILL_CHARACTER )
		ss << SEPARATOR << mFillCharacter;

	ss << CLOSE_BRACES;
	return ss.str();
}

void Expression::Variable::dump( ostream &io_stream ) const
{
	if ( !isValid() ) { io_stream << "KEY{invalid}"; return; }
	io_stream << "KEY_VAR{name:" << mName << ", minWidth:" << mMinWidth << ", fillChar:" << mFillCharacter << '}';
}

bool Expression::Variable::isVariable() const { return true; }

#ifdef __DEBUG_EXPRESSION
	Expression::Variable::Variable( const Variable &b ):Element(b)
	{
		__allocating("Expression::Variable::Variable(const Variable &)");
		mName = b.mName;
		mMinWidth = b.mMinWidth;
		mFillCharacter = b.mFillCharacter;
		mIsValid = b.mIsValid;
	}

	Expression::Variable::~Variable()
	{
		__deleting( "Expression::Variable::~Variable" );
	}
#endif


//----------------------------------------------------------------------
// methods of class Expression
//----------------------------------------------------------------------

// returns the length of the variable's name without decoration or string::npos if the syntax is invalid
static size_t get_variable_name_size( const char *i_str, const size_t i_length )
{
	if ( i_length<3 || i_str[0]!=VARIABLE_TAG || i_str[1]!=OPEN_BRACES ) return string::npos;
	for ( size_t i=2; i<i_length; i++ )
		if ( i_str[i]==CLOSE_BRACES ) return i-2;
	return string::npos;
}

void Expression::set( const string &i_str )
{
	clear();
	mIsValid = false;

	const size_t length = i_str.length();
	const char *c_str = i_str.c_str();
	size_t i = 0;
	while (  i<length )
	{
		if ( i_str[i]==VARIABLE_TAG )
		{
			size_t varLength = get_variable_name_size( c_str+i, length-i );
			if ( varLength==string::npos ) return;
			Variable *newVar = new Variable( i_str.substr(i+2,varLength) );

			if ( !newVar->isValid() )
			{
				delete newVar;
				return;
			}

			mElements.push_back( newVar );
			i += varLength+3;
		}
		else
		{
			size_t pos = i_str.find( VARIABLE_TAG, i );
			if ( pos==string::npos )
				mElements.push_back( new String( i_str.substr(i) ) );
			else
				mElements.push_back( new String( i_str.substr(i, pos-i) ) );
			i = pos;
		}
	}

	mIsValid = true;
}

void Expression::clear()
{
	list<Element*>::iterator it = mElements.begin();
	while ( it!=mElements.end() )
		delete *it++;
}

void Expression::set( const Expression &i_b )
{
	clear();

	// duplicate elements list
	list<Expression::Element *>::const_iterator it = i_b.mElements.begin();
	while ( it!=i_b.mElements.end() )
		mElements.push_back( Expression::Element::duplicate( **it++ ) );

	mIsValid = i_b.mIsValid;
}

void Expression::merge_subsequent_strings()
{
	if ( mElements.size()<2 ) return;
	list<Element*>::iterator it0 = mElements.begin(), it1 = it0;
	it1++;
	while ( it1!=mElements.end() )
	{
		if ( !(**it0).isVariable() && !(**it1).isVariable() )
		{
			TO_CHILD(String,**it0).mValue.append( TO_CHILD(String,**it1).mValue );
			delete *it1;
			it1 = mElements.erase(it1);
		}
		else
		{
			it0 = it1;
			it1++;
		}
	}
}

void Expression::dump( std::ostream &io_stream ) const
{
	cout << "expression = {" << endl;
	list<Element*>::const_iterator itElement = mElements.begin();
	while ( itElement!=mElements.end() )
	{
		cout << '\t' << (**itElement++) << endl;
	}
	cout << '}' << endl;
}

string Expression::toString() const
{
	string res;
	list<Element*>::const_iterator it = mElements.begin();
	while ( it!=mElements.end() )
		res.append( (**it++).toString() );
	return res;
}

bool Expression::hasVariable( const std::string &i_variableName ) const
{
	list<Element*>::const_iterator it = mElements.begin();
	while ( it!=mElements.end() )
	{
		const Element &e = **it++;
		if ( e.isVariable() && TO_CHILD(Variable,e).name()==i_variableName ) return true;
	}
	return false;
}

bool Expression::hasVariables( const list<string> &i_variableNames ) const
{
	list<string>::const_iterator it = i_variableNames.begin();
	while ( it!=i_variableNames.end() )
		if ( !hasVariable(*it++) ) return false;
	return true;
}
