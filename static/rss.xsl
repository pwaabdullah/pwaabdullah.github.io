<?xml version="1.0" encoding="UTF-8"?>
<!--
  rss.xsl — stylesheet applied to /index.xml (and other RSS feeds) so
  that when a human opens the feed URL in a browser, they see a clean
  styled page (with subscribe instructions + recent posts) instead of
  raw XML. Feed readers ignore this stylesheet entirely; they still
  parse the underlying RSS 2.0 XML.
-->
<xsl:stylesheet version="1.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:atom="http://www.w3.org/2005/Atom">
  <xsl:output method="html" version="1.0" encoding="UTF-8" indent="yes"/>

  <xsl:template match="/rss/channel">
    <html lang="en">
      <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1"/>
        <meta name="robots" content="noindex"/>
        <title>RSS feed — <xsl:value-of select="title"/></title>
        <link rel="icon" href="/favicon.ico"/>
        <style>
          :root { --brand: #00BFFF; --bg: #1d1e20; --fg: #e6e6e6; --muted: #9b9c9d; --card: #2e2e33; --border: #333; }
          @media (prefers-color-scheme: light) { :root { --bg: #fff; --fg: #1a1a1a; --muted: #555; --card: #f5f5f7; --border: #e5e5e5; } }
          body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif; background: var(--bg); color: var(--fg); margin: 0; padding: 2rem 1rem; line-height: 1.6; }
          .wrap { max-width: 760px; margin: 0 auto; }
          .badge { display: inline-block; background: var(--brand); color: #0F172A; padding: 0.15rem 0.6rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; letter-spacing: 0.04em; }
          h1 { font-size: 1.75rem; margin: 0.6rem 0 0.4rem; }
          .lede { color: var(--muted); margin: 0 0 1.5rem; }
          .info { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.2rem; margin: 1.5rem 0; }
          .info p { margin: 0.4rem 0; }
          code { background: var(--border); padding: 0.12rem 0.4rem; border-radius: 4px; font-size: 0.88em; }
          h2 { font-size: 1.15rem; margin: 2rem 0 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
          ul { list-style: none; padding: 0; margin: 0; }
          li { border-bottom: 1px solid var(--border); padding: 0.9rem 0; }
          li:last-child { border: 0; }
          .item-title { font-weight: 600; font-size: 1.02rem; }
          .item-title a { color: var(--fg); text-decoration: none; }
          .item-title a:hover { color: var(--brand); text-decoration: underline; }
          .item-meta { color: var(--muted); font-size: 0.85rem; margin-top: 0.2rem; }
          .item-desc { color: var(--muted); margin-top: 0.4rem; font-size: 0.95rem; }
          a { color: var(--brand); }
          .back { margin-top: 2rem; display: inline-block; }
        </style>
      </head>
      <body>
        <div class="wrap">
          <span class="badge">RSS feed</span>
          <h1><xsl:value-of select="title"/></h1>
          <p class="lede"><xsl:value-of select="description"/></p>

          <div class="info">
            <p><strong>This is an RSS feed.</strong> Subscribe in your favorite reader to get new posts automatically.</p>
            <p>Just copy this page's URL into a feed reader (NetNewsWire, Feedly, Reeder, Inoreader, Thunderbird, etc.).</p>
          </div>

          <h2>Recent items</h2>
          <ul>
            <xsl:for-each select="item">
              <li>
                <div class="item-title">
                  <a hreflang="en">
                    <xsl:attribute name="href"><xsl:value-of select="link"/></xsl:attribute>
                    <xsl:value-of select="title"/>
                  </a>
                </div>
                <div class="item-meta">
                  <xsl:value-of select="pubDate"/>
                </div>
                <xsl:if test="description">
                  <div class="item-desc">
                    <xsl:value-of select="description" disable-output-escaping="no"/>
                  </div>
                </xsl:if>
              </li>
            </xsl:for-each>
          </ul>

          <a class="back" href="/">&#8592; Back to site</a>
        </div>
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
