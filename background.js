/**
 * Anywear VTO - Background Service Worker (Manifest V3)
 * Handles:
 *  1. Persistent toolbar Action toggle (opens / collapses floating widget).
 *  2. Right-click Context Menu integration ("Try on this garment with Anywear VTO").
 *  3. Dynamic content script injection across open tabs on install/reload.
 */

const CONTEXT_MENU_ID = "anywear_vto_try_on";

// Domains excluded from automatic script injection and webcam operations
const EXCLUDED_HOSTS = [
  "gemini.google.com",
  "github.com",
  "google.com",
  "www.google.com",
  "mail.google.com",
  "drive.google.com",
  "docs.google.com",
  "stackoverflow.com",
  "chatgpt.com",
  "claude.ai"
];

function isExcludedUrl(url) {
  if (!url) return true;
  try {
    const parsed = new URL(url);
    if (parsed.protocol === "chrome:" || parsed.protocol === "edge:" || parsed.protocol === "about:" || parsed.protocol === "chrome-extension:") {
      return true;
    }
    const host = parsed.hostname.toLowerCase();
    return EXCLUDED_HOSTS.some(excluded => host === excluded || host.endsWith("." + excluded));
  } catch {
    return true;
  }
}

// Create Context Menu on install
chrome.runtime.onInstalled.addListener(async () => {
  console.log("[Anywear VTO] Extension initialized / updated.");

  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: CONTEXT_MENU_ID,
      title: "Try on this garment with Anywear VTO",
      contexts: ["image"]
    });
  });

  // Inject content scripts into already-open valid tabs on install/reload (skipping excluded domains)
  try {
    const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*", "file://*/*"] });
    for (const tab of tabs) {
      if (tab.id && !isExcludedUrl(tab.url)) {
        try {
          await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            files: ["content.js"]
          });
          await chrome.scripting.insertCSS({
            target: { tabId: tab.id },
            files: ["content.css"]
          });
        } catch {
          // Tab might be discarded or protected, ignore safely
        }
      }
    }
  } catch (err) {
    console.debug("[Anywear VTO] Tab query on install error:", err);
  }
});

// Toolbar Action Clicked (Toggle Overlay Modal)
chrome.action.onClicked.addListener(async (tab) => {
  if (!tab.id) return;
  if (tab.url && (tab.url.startsWith("chrome://") || tab.url.startsWith("edge://") || tab.url.startsWith("chrome-extension://"))) {
    console.warn("[Anywear VTO] Cannot operate on internal browser pages.");
    return;
  }

  await ensureAndSendMessage(tab.id, { action: "TOGGLE_MODAL" });
});

// Right-Click Context Menu Clicked
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === CONTEXT_MENU_ID && info.srcUrl && tab && tab.id) {
    console.log("[Anywear VTO] Context menu clicked on image URL:", info.srcUrl);
    await ensureAndSendMessage(tab.id, {
      action: "TRY_ON_IMAGE",
      srcUrl: info.srcUrl
    });
  }
});

/**
 * Safely sends message to tab, injecting scripts if tab was loaded before extension
 */
function ensureAndSendMessage(tabId, message) {
  if (!chrome.runtime?.id) return;

  try {
    chrome.tabs.sendMessage(tabId, message, (response) => {
      // Always inspect lastError to prevent Chrome from flagging an uncaught rejection
      const err = chrome.runtime.lastError;
      if (err) {
        // Tab does not have content script yet, inject dynamically and deliver
        try {
          chrome.scripting.executeScript({
            target: { tabId },
            files: ["content.js"]
          }, () => {
            if (chrome.runtime?.lastError) return;
            chrome.scripting.insertCSS({
              target: { tabId },
              files: ["content.css"]
            }, () => {
              if (chrome.runtime?.lastError) return;
              chrome.tabs.sendMessage(tabId, message, () => {
                // Consume lastError to avoid uncaught exception badge
                const _ignored = chrome.runtime?.lastError;
              });
            });
          });
        } catch {
          // Tab closed or protected
        }
      }
    });
  } catch {
    // Context invalidated or tab closed
  }
}
