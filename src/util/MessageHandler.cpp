#include "general/MessageHandler.h"

using namespace std;

//----------------------------------------------------------------------
// ListMessageHandler methods
//----------------------------------------------------------------------

ListMessageHandler::~ListMessageHandler()
{
	for ( list<Message*>::iterator it=m_messages.begin(); it!=m_messages.end(); it++ )
		delete *it;
}

//----------------------------------------------------------------------
// OStreamMessageHandler methods
//----------------------------------------------------------------------

void OStreamMessageHandler::add( const Message &i_message )
{
	m_stream << m_prefix << i_message.toString() << std::endl;
	if (m_action==NOTHING) return;
	else if (m_action==EXIT) exit(m_exitCode);
	else if (m_action==CIN_GET) std::cin.get();
}

//----------------------------------------------------------------------
// related functions
//----------------------------------------------------------------------

string eToString(MessageHandler::eAction e)
{
	switch (e)
	{
	case MessageHandler::CIN_GET: return "cin_get";
	case MessageHandler::EXIT: return "exit";
	case MessageHandler::NOTHING: return "nothing";
	}
	return "unknown";
}
