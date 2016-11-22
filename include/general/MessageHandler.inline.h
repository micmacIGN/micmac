// this file should be included by Expression.h solely

//----------------------------------------------------------------------
// methods of class Message
//----------------------------------------------------------------------

inline Message::~Message(){}


//----------------------------------------------------------------------
// methods of class StringMessage
//----------------------------------------------------------------------

inline StringMessage::StringMessage( const std::string &i_value ):m_value(i_value){}

inline std::string StringMessage::toString() const { return m_value; }

inline Message * StringMessage::duplicate() const { return new StringMessage(*this); }


//----------------------------------------------------------------------
// methods of class DebugErrorMessage
//----------------------------------------------------------------------

inline DebugErrorMessage::DebugErrorMessage( const std::string &i_file, int i_line, const std::string &i_where, const std::string &i_what ):
	m_file( i_file ),
	m_line( i_line ),
	m_where( i_where ),
	m_what( i_what ){}

inline std::string DebugErrorMessage::toString() const
{
	std::stringstream ss;
	ss << m_file << ':' << m_line << ' ' << m_where << ": " << m_what;
	return ss.str();
}

inline Message * DebugErrorMessage::duplicate() const { return new DebugErrorMessage(*this); }


//----------------------------------------------------------------------
// methods of class MessageHandler
//----------------------------------------------------------------------

inline void MessageHandler::setAction( eAction i_action, int i_exitCode )
{
	m_action = i_action;
	m_exitCode = i_exitCode;
}

inline MessageHandler::MessageHandler( eAction i_action, int i_exitCode ){ setAction(i_action,i_exitCode); }

inline MessageHandler::~MessageHandler(){}

inline MessageHandler::eAction MessageHandler::action() const { return m_action; }

inline int MessageHandler::exitCode() const { return m_exitCode; }


//----------------------------------------------------------------------
// methods of class ListMessageHandler
//----------------------------------------------------------------------

inline ListMessageHandler::ListMessageHandler(){}

inline void ListMessageHandler::add( const Message &i_message ) { m_messages.push_back( i_message.duplicate() ); }


//----------------------------------------------------------------------
// methods of class OStreamMessageHandler
//----------------------------------------------------------------------

inline OStreamMessageHandler::~OStreamMessageHandler(){}

inline OStreamMessageHandler::OStreamMessageHandler( std::ostream &io_ostream, const std::string &i_prefix, eAction i_action, int i_exitCode ):
	MessageHandler(i_action,i_exitCode),
	m_stream(io_ostream),
	m_prefix(i_prefix){}
