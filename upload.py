"""
Upload and update Notion pages.
"""

import os
import sys

from notion.client import NotionClient
from notion.block import PageBlock, TextBlock, HeaderBlock, SubheaderBlock

class Uploader:
    def __init__(self):
        # Obtained by inspecting browser cookies and stored in env.
        notion_token = os.environ['NOTION_TOKEN']
        notion_url = os.environ['NOTION_URL']
        assert notion_token
        assert notion_url

        self.client = NotionClient(token_v2=notion_token)
        self.page = self.client.get_block(notion_url)

    def upload(self, title, summary, text, datetime):
        """Upload the text data as a new child page."""
        page = self.page.children.add_new(PageBlock, title=title)

        page.children.add_new(SubheaderBlock, title='Summary')
        page.children.add_new(TextBlock, title='Date: {}'.format(datetime))
        page.children.add_new(TextBlock, title=summary)
        
        page.children.add_new(SubheaderBlock, title='Transcript')
        page.children.add_new(TextBlock, title=text)