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
