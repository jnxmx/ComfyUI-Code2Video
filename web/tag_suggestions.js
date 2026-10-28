/**
 * Tag suggestions helper for @media mentions in prompt text inputs.
 */

export function attachTagSuggestions(textarea, getAvailableTags) {
    if (!textarea) return;

    let popup = document.querySelector(".c2v-tag-popup");
    if (!popup) {
        popup = document.createElement("div");
        popup.className = "c2v-tag-popup";
        document.body.appendChild(popup);
    }

    let activeIndex = -1;
    let matchStart = -1;

    function hidePopup() {
        popup.style.display = "none";
        activeIndex = -1;
        matchStart = -1;
    }

    function showPopup(tags, left, top) {
        popup.innerHTML = "";
        tags.forEach((tag, idx) => {
            const item = document.createElement("div");
            item.className = `c2v-tag-item ${idx === activeIndex ? "active" : ""}`;
            item.textContent = tag;
            item.addEventListener("mousedown", (e) => {
                e.preventDefault();
                insertTag(tag);
            });
            popup.appendChild(item);
        });

        popup.style.left = `${left}px`;
        popup.style.top = `${top}px`;
        popup.style.display = "block";
    }

    function insertTag(tag) {
        const val = textarea.value;
        const cursor = textarea.selectionStart;
        const before = val.slice(0, matchStart);
        const after = val.slice(cursor);
        textarea.value = before + tag + " " + after;
        textarea.selectionStart = textarea.selectionEnd = matchStart + tag.length + 1;
        hidePopup();
        textarea.dispatchEvent(new Event("input"));
    }

    textarea.addEventListener("input", () => {
        const cursor = textarea.selectionStart;
        const textBefore = textarea.value.slice(0, cursor);
        const lastAt = textBefore.lastIndexOf("@");

        if (lastAt !== -1 && !/\s/.test(textBefore.slice(lastAt + 1))) {
            const query = textBefore.slice(lastAt + 1).toLowerCase();
            const available = getAvailableTags ? getAvailableTags() : ["@image_1", "@image_2", "@video_1", "@audio_1"];
            const filtered = available.filter(t => t.toLowerCase().includes(query));

            if (filtered.length > 0) {
                matchStart = lastAt;
                const rect = textarea.getBoundingClientRect();
                showPopup(filtered, rect.left + 10, rect.bottom + window.scrollY);
                return;
            }
        }
        hidePopup();
    });

    textarea.addEventListener("keydown", (e) => {
        if (popup.style.display === "block") {
            const items = popup.querySelectorAll(".c2v-tag-item");
            if (e.key === "ArrowDown") {
                e.preventDefault();
                activeIndex = (activeIndex + 1) % items.length;
                items.forEach((it, i) => it.classList.toggle("active", i === activeIndex));
            } else if (e.key === "ArrowUp") {
                e.preventDefault();
                activeIndex = (activeIndex - 1 + items.length) % items.length;
                items.forEach((it, i) => it.classList.toggle("active", i === activeIndex));
            } else if (e.key === "Enter" || e.key === "Tab") {
                if (activeIndex >= 0 && items[activeIndex]) {
                    e.preventDefault();
                    insertTag(items[activeIndex].textContent);
                }
            } else if (e.key === "Escape") {
                hidePopup();
            }
        }
    });

    textarea.addEventListener("blur", () => {
        setTimeout(hidePopup, 150);
    });
}
