/**
 * Anywear VTO - Background Service Worker (Manifest V3)
 * Handles:
 *  1. Persistent toolbar Action toggle (opens / collapses floating widget).
 *  2. Right-click Context Menu integration ("Try on this garment with Anywear VTO").
 *  3. Dynamic content script injection across open tabs on install/reload.
 */

const CONTEXT_MENU_ID = "anywear_vto_try_on";

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

  // Inject content scripts into already-open valid tabs on install/reload
  try {
    const tabs = await chrome.tabs.query({ url: ["http://*/*", "https://*/*", "file://*/*"] });
    for (const tab of tabs) {
      if (tab.id && !tab.url.startsWith("chrome://") && !tab.url.startsWith("edge://")) {
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
async function ensureAndSendMessage(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch (err) {
    console.log("[Anywear VTO] Injecting content script dynamically to deliver message...", err);
    try {
      await chrome.scripting.executeScript({
        target: { tabId },
        files: ["content.js"]
      });
      await chrome.scripting.insertCSS({
        target: { tabId },
        files: ["content.css"]
      });
      // Retry sending message after brief yield
      setTimeout(async () => {
        try {
          await chrome.tabs.sendMessage(tabId, message);
        } catch (retryErr) {
          console.warn("[Anywear VTO] Could not deliver message after dynamic injection:", retryErr);
        }
      }, 100);
    } catch (injectErr) {
      console.error("[Anywear VTO] Failed dynamic script injection:", injectErr);
    }
  }
}
